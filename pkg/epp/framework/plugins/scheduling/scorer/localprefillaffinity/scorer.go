/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package localprefillaffinity provides a scorer for the prefill scheduling profile that
// boosts the decode-selected pod (a flex/kv_both pod) when the non-cached portion of the
// prompt is short enough that local prefill+decode is faster than disaggregated P/D.
//
// Background: P/D disaggregation incurs a fixed overhead (~44ms on H200) from inter-pod
// network hops and KV-cache bootstrap coordination. For short (uncached) prompts the cold
// prefill compute is cheaper than that overhead, so routing to the same pod that will do
// decode avoids the x-prefiller-host-port header entirely (detected by PreRequest).
//
// The crossover threshold (localBonusTokens) is auto-calibrated ONCE per deployment from
// Prometheus metrics, because it depends only on the hardware:
//
//	localBonusTokens = pdOverheadMs / ms_per_tok_cold
//
// where ms_per_tok_cold is measured per-engine from the prefill pod's metrics. Calibration
// runs once the first minSamples prefill requests have been recorded, then stops.
//
// TODO(future — P2P KV cache sharing): When llm-d gains peer-to-peer KV cache sharing
// between the dedicated prefill pod and the flex pod, this scorer should also consider the
// prefill pod's cache hits. If a remote prefill pod already holds a warm prefix that the
// flex pod lacks, disaggregating could be cheaper even for short prompts, because the
// prefill pod could stream just the newly-computed suffix (and share its hit) to the flex
// pod over the P2P channel. At that point the decision becomes:
//
//	local_cost = nonCachedTokens_flex * ms_per_tok_cold
//	pd_cost    = pdOverheadMs + nonCachedTokens_prefill * ms_per_tok_cold
//
// and we route local only when local_cost < pd_cost, using each pod's own cache-hit state
// rather than assuming the flex pod always recomputes the full uncached suffix.
package localprefillaffinity

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	// Type is the canonical plugin type string for this scorer.
	Type = "local-prefill-affinity-scorer"

	// DecodeEndpointKey is the request-attribute key under which the disagg profile
	// handler stores the decode-selected endpoint before running the prefill profile.
	// The scorer reads this to know which candidate to boost.
	DecodeEndpointKey = "local-prefill-affinity/decode-endpoint"

	defaultLocalBonusTokens int64   = 400
	defaultPDOverheadMs     float64 = 44
	defaultMinSamples       int     = 10
	defaultCalibrationPoll          = 30 * time.Second
)

// compile-time interface assertion
var _ fwksched.Scorer = &Scorer{}

// parameters is the JSON config for this scorer.
type parameters struct {
	// PrometheusURL is the base URL of the Prometheus HTTP API (e.g. "http://prometheus:9090").
	// When empty, auto-calibration is disabled and localBonusTokens stays at the default.
	PrometheusURL string `json:"prometheusURL"`

	// PDOverheadMs is the total P/D disaggregation overhead in milliseconds — the flat cost
	// of an extra network hop + KV bootstrap, measured as warm-cache P/D TTFT. Default 44ms
	// (measured on H200 with 120B model). This value is cluster-topology-specific; tune it
	// if your intra-cluster latency differs significantly.
	PDOverheadMs float64 `json:"pdOverheadMs"`

	// DefaultLocalBonusTokens is the initial threshold used before calibration completes
	// (or when auto-calibration is disabled). Default 400 tokens.
	DefaultLocalBonusTokens int64 `json:"defaultLocalBonusTokens"`

	// CalibrationPollSeconds is how often (in seconds) to poll Prometheus while waiting for
	// the first minSamples requests to accumulate. Once calibration succeeds it runs no more.
	// Default 30.
	CalibrationPollSeconds int `json:"calibrationPollSeconds"`

	// MinSamples is the minimum number of prefill requests that must have been recorded
	// before the one-time calibration is trusted. Default 10.
	MinSamples int `json:"minSamples"`
}

// Scorer boosts the decode-selected pod during prefill profile scoring when the non-cached
// prompt is shorter than localBonusTokens, steering the system toward local execution.
type Scorer struct {
	typedName        fwkplugin.TypedName
	localBonusTokens atomic.Int64
	pdOverheadMs     float64
	minSamples       int
	prometheusURL    string
}

// Factory is the plugin factory function registered with the EPP plugin registry.
func Factory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p := parameters{
		PDOverheadMs:            defaultPDOverheadMs,
		DefaultLocalBonusTokens: defaultLocalBonusTokens,
		CalibrationPollSeconds:  int(defaultCalibrationPoll.Seconds()),
		MinSamples:              defaultMinSamples,
	}
	if decoder != nil {
		if err := decoder.Decode(&p); err != nil {
			return nil, fmt.Errorf("failed to decode %s parameters: %w", Type, err)
		}
	}

	s := &Scorer{
		typedName:     fwkplugin.TypedName{Type: Type, Name: name},
		pdOverheadMs:  p.PDOverheadMs,
		minSamples:    p.MinSamples,
		prometheusURL: p.PrometheusURL,
	}
	s.localBonusTokens.Store(p.DefaultLocalBonusTokens)

	if p.PrometheusURL != "" {
		poll := time.Duration(p.CalibrationPollSeconds) * time.Second
		// One-time calibration: localBonusTokens is a function of the hardware only,
		// so we tune it once at deployment start and then leave it fixed.
		go s.calibrateOnce(handle.Context(), poll)
	}

	return s, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *Scorer) TypedName() fwkplugin.TypedName { return s.typedName }

// Category returns Affinity — this scorer gives a hard preference to one endpoint.
func (s *Scorer) Category() fwksched.ScorerCategory { return fwksched.Affinity }

// Score gives 1.0 to the decode-selected endpoint if it is also in the prefill candidate
// list and the non-cached prompt is shorter than localBonusTokens; all other endpoints get
// 0.0. When the decode endpoint is unknown or the non-cached suffix is long, all scores are
// 0.0, leaving the decision to other scorers (e.g. prefix-cache affinity).
func (s *Scorer) Score(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))

	decodeEp, ok := fwksched.ReadRequestAttribute[fwksched.Endpoint](req, DecodeEndpointKey)
	if !ok {
		return scores
	}

	inputTokens := inputTokenCount(req)
	if inputTokens == 0 {
		return scores
	}
	bonus := s.localBonusTokens.Load()
	decodeNSN := decodeEp.GetMetadata().NamespacedName

	for _, ep := range endpoints {
		if ep.GetMetadata().NamespacedName != decodeNSN {
			continue
		}
		// Compare the NON-cached suffix (what we'd actually recompute on the flex pod),
		// not the raw prompt length — cache hits are free and must not count against
		// local execution. This mirrors the prefix-based PD decider and matches the
		// calibration formula, whose ms_per_tok was measured against computed tokens.
		nonCached := nonCachedTokens(ep, inputTokens)
		if nonCached > 0 && nonCached < int(bonus) {
			scores[ep] = 1.0
			log.FromContext(ctx).V(4).Info("local-prefill-affinity: boosting flex pod",
				"pod", decodeNSN, "nonCachedTokens", nonCached, "threshold", bonus)
		}
	}
	return scores
}

// inputTokenCount returns the tokenized prompt length, or 0 if unavailable.
func inputTokenCount(req *fwksched.InferenceRequest) int {
	if req == nil || req.Body == nil || req.Body.TokenizedPrompt == nil {
		return 0
	}
	return req.Body.TokenizedPrompt.TokenCount()
}

// nonCachedTokens returns the number of prompt tokens the endpoint would have to compute
// (input minus the contiguous cached prefix it already holds). Falls back to the full input
// length when the endpoint has no prefix-cache match info attached.
func nonCachedTokens(ep fwksched.Endpoint, inputTokens int) int {
	raw, ok := ep.Get(attrprefix.PrefixCacheMatchInfoDataKey.String())
	if !ok || raw == nil {
		return inputTokens
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		return inputTokens
	}
	hitPrefixTokens := info.CachedBlockCount() * info.BlockSizeTokens()
	nc := inputTokens - hitPrefixTokens
	if nc < 0 {
		return 0
	}
	return nc
}

// ── One-time Prometheus-based calibration ─────────────────────────────────────

// calibrateOnce polls Prometheus until the first minSamples prefill requests have been
// recorded, computes localBonusTokens from the hardware's cold-prefill speed, stores it,
// and returns. It exits early if ctx is cancelled. Because the threshold depends only on
// hardware, there is no need to re-run it for the life of the deployment.
func (s *Scorer) calibrateOnce(ctx context.Context, poll time.Duration) {
	logger := log.FromContext(ctx).WithName("local-prefill-affinity-calibration")
	ticker := time.NewTicker(poll)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			bonus, err := s.calibrate(ctx)
			if err != nil {
				logger.V(2).Info("waiting for calibration samples", "reason", err.Error())
				continue
			}
			old := s.localBonusTokens.Swap(bonus)
			logger.Info("localBonusTokens calibrated (one-time)",
				"previous", old, "calibrated", bonus, "pdOverheadMs", s.pdOverheadMs)
			return
		}
	}
}

// calibrate queries Prometheus and returns a new localBonusTokens value.
// It detects the engine type (vLLM vs SGLang) by which metric is present.
func (s *Scorer) calibrate(ctx context.Context) (int64, error) {
	// ── Try vLLM first ──────────────────────────────────────────────────────
	// Query prefill pod: request_prefill_time_seconds and request_prefill_kv_computed_tokens.
	// Cache hits contribute ~0 to both numerator and denominator, so the ratio converges
	// to cold-prefill speed automatically.
	pftSum, pftCount, err := s.querySum(ctx, "vllm:request_prefill_time_seconds_sum")
	if err == nil && pftCount >= float64(s.minSamples) {
		kvSum, _, err2 := s.querySum(ctx, "vllm:request_prefill_kv_computed_tokens_sum")
		if err2 == nil && kvSum > 0 {
			msPerTok := (pftSum / kvSum) * 1000
			return computeBonus(s.pdOverheadMs, msPerTok), nil
		}
	}

	// ── Try SGLang ──────────────────────────────────────────────────────────
	// prefill_forward stage latency from the prefill pod, normalized by uncached tokens
	// from the decode pod's histogram.
	pfwdSum, pfwdCount, err := s.queryStageSum(ctx,
		"sglang_per_stage_req_latency_seconds_sum", "prefill_forward")
	if err == nil && pfwdCount >= float64(s.minSamples) {
		uncSum, uncCount, err2 := s.querySum(ctx, "sglang_uncached_prompt_tokens_histogram_sum")
		if err2 == nil && uncCount > 0 && uncSum > 0 {
			avgForwardMs := (pfwdSum / pfwdCount) * 1000
			avgUncachedToks := uncSum / uncCount
			msPerTok := avgForwardMs / avgUncachedToks
			return computeBonus(s.pdOverheadMs, msPerTok), nil
		}
	}

	return 0, fmt.Errorf("insufficient samples (vLLM count=%.0f, SGLang count=%.0f, need %d)",
		pftCount, pfwdCount, s.minSamples)
}

// computeBonus computes localBonusTokens from the P/D overhead and cold prefill speed.
// Returns the default if inputs are invalid.
func computeBonus(pdOverheadMs, msPerTok float64) int64 {
	if msPerTok <= 0 {
		return defaultLocalBonusTokens
	}
	v := math.Round(pdOverheadMs / msPerTok)
	if v < 1 {
		return 1
	}
	if v > math.MaxInt32 {
		return defaultLocalBonusTokens
	}
	return int64(v)
}

// ── Prometheus HTTP query helpers ─────────────────────────────────────────────

type promResponse struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string `json:"resultType"`
		Result     []struct {
			Metric map[string]string  `json:"metric"`
			Value  [2]json.RawMessage `json:"value"`
		} `json:"result"`
	} `json:"data"`
}

// querySum fetches the current value of a Prometheus instant query (scalar or vector).
// Returns (sum, count, error) where count is the number of result series returned.
func (s *Scorer) querySum(ctx context.Context, metric string) (sum, count float64, err error) {
	results, err := s.promQuery(ctx, metric)
	if err != nil {
		return 0, 0, err
	}
	for _, r := range results {
		var v float64
		if err := json.Unmarshal(r.Value[1], &v); err != nil {
			continue
		}
		sum += v
		count++
	}
	if count == 0 {
		return 0, 0, fmt.Errorf("metric %s returned no data", metric)
	}
	return sum, count, nil
}

// queryStageSum fetches the summed value and series count for a per-stage histogram metric,
// filtered by stage label. Returns (totalValue, seriesCount); the caller divides to get an
// average.
func (s *Scorer) queryStageSum(ctx context.Context, metric, stage string) (sum, count float64, err error) {
	q := fmt.Sprintf(`%s{stage="%s"}`, metric, stage)
	results, err := s.promQuery(ctx, q)
	if err != nil {
		return 0, 0, err
	}
	for _, r := range results {
		var v float64
		if err := json.Unmarshal(r.Value[1], &v); err != nil {
			continue
		}
		sum += v
		count++
	}
	if count == 0 {
		return 0, 0, fmt.Errorf("metric %s{stage=%q} returned no data", metric, stage)
	}
	return sum, count, nil
}

// promQuery executes a Prometheus instant query and returns the raw result slice.
func (s *Scorer) promQuery(ctx context.Context, q string) ([]struct {
	Metric map[string]string  `json:"metric"`
	Value  [2]json.RawMessage `json:"value"`
}, error) {
	u := fmt.Sprintf("%s/api/v1/query?query=%s", s.prometheusURL, url.QueryEscape(q))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var pr promResponse
	if err := json.Unmarshal(body, &pr); err != nil {
		return nil, err
	}
	if pr.Status != "success" {
		return nil, fmt.Errorf("prometheus query failed: status=%s", pr.Status)
	}
	return pr.Data.Result, nil
}
