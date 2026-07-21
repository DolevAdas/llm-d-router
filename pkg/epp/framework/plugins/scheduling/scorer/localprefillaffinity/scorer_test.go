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

package localprefillaffinity

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// makeEndpoint builds an endpoint with the given name and optional cached-prefix tokens.
// When cachedTokens < 0, no PrefixCacheMatchInfo is attached (simulating a missing producer).
func makeEndpoint(name string, cachedTokens int) fwksched.Endpoint {
	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: name},
			Address:        "10.0.0.1",
			Port:           "8000",
		},
		nil,
		fwkdl.NewAttributes(),
	)
	if cachedTokens >= 0 {
		// blockSizeTokens=1 so cachedBlockCount == cachedTokens.
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(),
			attrprefix.NewPrefixCacheMatchInfo(cachedTokens, 10000, 1))
	}
	return ep
}

// makeRequest builds a request whose tokenized prompt carries promptTokens token IDs,
// optionally storing decodeEp under DecodeEndpointKey.
func makeRequest(promptTokens int, decodeEp fwksched.Endpoint) *fwksched.InferenceRequest {
	req := &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{make([]uint32, promptTokens)},
			},
		},
	}
	if decodeEp != nil {
		req.PutAttribute(DecodeEndpointKey, decodeEp)
	}
	return req
}

func newScorer(bonus int64) *Scorer {
	s := &Scorer{
		typedName:    fwkplugin.TypedName{Type: Type, Name: Type},
		pdOverheadMs: defaultPDOverheadMs,
		minSamples:   defaultMinSamples,
	}
	s.localBonusTokens.Store(bonus)
	return s
}

func TestScore(t *testing.T) {
	const bonus = 400

	flex := makeEndpoint("flex-pod", -1)  // no cache info → full input is non-cached
	other := makeEndpoint("other-pod", -1)

	tests := []struct {
		name         string
		promptTokens int
		decodeEp     fwksched.Endpoint
		candidates   []fwksched.Endpoint
		wantBoosted  map[string]bool // endpoint name → expected score 1.0
	}{
		{
			name:         "short prompt boosts the decode pod",
			promptTokens: 100,
			decodeEp:     flex,
			candidates:   []fwksched.Endpoint{flex, other},
			wantBoosted:  map[string]bool{"flex-pod": true},
		},
		{
			name:         "long prompt does not boost (prefer P/D)",
			promptTokens: 800,
			decodeEp:     flex,
			candidates:   []fwksched.Endpoint{flex, other},
			wantBoosted:  map[string]bool{},
		},
		{
			name:         "prompt exactly at threshold does not boost",
			promptTokens: bonus,
			decodeEp:     flex,
			candidates:   []fwksched.Endpoint{flex, other},
			wantBoosted:  map[string]bool{},
		},
		{
			name:         "no decode endpoint attribute → no scores",
			promptTokens: 100,
			decodeEp:     nil,
			candidates:   []fwksched.Endpoint{flex, other},
			wantBoosted:  map[string]bool{},
		},
		{
			name:         "decode pod not among prefill candidates → no boost",
			promptTokens: 100,
			decodeEp:     makeEndpoint("absent-pod", -1),
			candidates:   []fwksched.Endpoint{flex, other},
			wantBoosted:  map[string]bool{},
		},
		{
			name:         "zero prompt tokens → no scores",
			promptTokens: 0,
			decodeEp:     flex,
			candidates:   []fwksched.Endpoint{flex, other},
			wantBoosted:  map[string]bool{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := newScorer(bonus)
			req := makeRequest(tc.promptTokens, tc.decodeEp)
			scores := s.Score(context.Background(), req, tc.candidates)

			for _, ep := range tc.candidates {
				name := ep.GetMetadata().NamespacedName.Name
				score, present := scores[ep]
				if tc.wantBoosted[name] {
					assert.True(t, present && score == 1.0,
						"endpoint %s should be boosted to 1.0, got %v (present=%v)", name, score, present)
				} else {
					assert.False(t, present && score > 0,
						"endpoint %s should not be boosted, got %v", name, score)
				}
			}
		})
	}
}

// TestScoreUsesNonCachedTokens verifies that cached prefix tokens are subtracted before
// comparing against the threshold — a long prompt with a large cache hit still routes local.
func TestScoreUsesNonCachedTokens(t *testing.T) {
	const bonus = 400

	// 1000-token prompt, but 800 tokens already cached on the flex pod → 200 non-cached < 400.
	flex := makeEndpoint("flex-pod", 800)
	s := newScorer(bonus)
	req := makeRequest(1000, flex)

	scores := s.Score(context.Background(), req, []fwksched.Endpoint{flex})
	assert.Equal(t, 1.0, scores[flex],
		"1000-token prompt with 800 cached (200 non-cached) should be boosted for local execution")

	// Same 1000-token prompt but only 100 cached → 900 non-cached >= 400 → prefer P/D.
	flexColdish := makeEndpoint("flex-pod", 100)
	req2 := makeRequest(1000, flexColdish)
	scores2 := s.Score(context.Background(), req2, []fwksched.Endpoint{flexColdish})
	_, present := scores2[flexColdish]
	assert.False(t, present, "1000-token prompt with only 100 cached (900 non-cached) should NOT be boosted")
}

func TestComputeBonus(t *testing.T) {
	tests := []struct {
		name         string
		pdOverheadMs float64
		msPerTok     float64
		want         int64
	}{
		{"vLLM H200 example", 44, 0.1570, 280},      // matches empirical benchmark
		{"fast hardware → lower threshold", 44, 1.0, 44},
		{"slow prefill → higher threshold", 44, 0.05, 880},
		{"zero msPerTok falls back to default", 44, 0, defaultLocalBonusTokens},
		{"negative msPerTok falls back to default", 44, -1, defaultLocalBonusTokens},
		{"rounds to at least 1", 1, 1000, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := computeBonus(tc.pdOverheadMs, tc.msPerTok)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestNonCachedTokens(t *testing.T) {
	// No prefix info attached → full input is non-cached.
	epNoInfo := makeEndpoint("p", -1)
	assert.Equal(t, 500, nonCachedTokens(epNoInfo, 500))

	// 300 cached out of 500 → 200 non-cached.
	epHit := makeEndpoint("p", 300)
	assert.Equal(t, 200, nonCachedTokens(epHit, 500))

	// More cached than input (stale/over-count) → clamped to 0.
	epOver := makeEndpoint("p", 800)
	assert.Equal(t, 0, nonCachedTokens(epOver, 500))
}
