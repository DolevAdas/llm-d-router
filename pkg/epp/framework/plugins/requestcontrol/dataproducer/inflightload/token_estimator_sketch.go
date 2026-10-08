/*
Copyright 2026 The llm-d Authors.

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

package inflightload

import (
	lru "github.com/hashicorp/golang-lru/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload/outlensketch"
)

// sketchPredictionCacheSize bounds the per-request prediction cache. A request's
// output estimate is endpoint-independent but EstimateOutputFromRequest is called
// once per candidate endpoint, so the first call per request is cached and the rest
// read it back. The bound keeps memory flat if an entry is never explicitly removed
// (e.g. a request that ends without an end-of-stream).
const sketchPredictionCacheSize = 8192

// OutputObserver is an optional TokenEstimator capability: a learning estimator
// implements it to receive the observed output length at end-of-stream. The
// producer calls it only when its estimator implements it, so the static estimator
// is unaffected.
type OutputObserver interface {
	Observe(request *fwksched.InferenceRequest, completionTokens int64)
}

// sketchModel is the output-length predictor and learner the estimator drives.
// *outlensketch.Sketch is the production implementation; the seam keeps the
// estimator's caching and fallback logic testable without the embedding asset.
type sketchModel interface {
	Predict(body *fwkrh.InferenceRequestBody) (magnitude int64, ok bool)
	Observe(body *fwkrh.InferenceRequestBody, completionTokens int64)
}

var _ sketchModel = (*outlensketch.Sketch)(nil)

// SketchTokenEstimator estimates output tokens from the learned output-length
// sketch (outlensketch): a more precise, self-adapting alternative to the static
// signal-rule mapping that an operator selects in place of it. Input-token
// estimation, output clamping, and the cold-key fallback reuse SimpleTokenEstimator.
type SketchTokenEstimator struct {
	*SimpleTokenEstimator
	sketch sketchModel
	cache  *lru.Cache[string, int64]
}

var (
	_ TokenEstimator = &SketchTokenEstimator{}
	_ OutputObserver = &SketchTokenEstimator{}
)

// NewSketchTokenEstimator returns a SketchTokenEstimator over the given sketch, with
// an optional operator cap on the estimated output tokens.
func NewSketchTokenEstimator(sketch sketchModel, maxOutput *int64) (*SketchTokenEstimator, error) {
	cache, err := lru.New[string, int64](sketchPredictionCacheSize)
	if err != nil {
		return nil, err
	}
	return &SketchTokenEstimator{
		SimpleTokenEstimator: &SimpleTokenEstimator{MaxEstimatedOutputTokens: maxOutput},
		sketch:               sketch,
		cache:                cache,
	}, nil
}

// EstimateOutputFromRequest returns the sketch's output-token estimate for the
// request, cached per request id so repeated per-endpoint calls embed the prompt
// once. A cold key falls back to the static signal-rule estimate, so the estimator
// is usable from the first request and self-heals as keys warm up.
func (e *SketchTokenEstimator) EstimateOutputFromRequest(request *fwksched.InferenceRequest) int64 {
	if request == nil || request.Body == nil {
		return 0
	}
	if request.RequestID != "" {
		if cached, ok := e.cache.Get(request.RequestID); ok {
			return cached
		}
	}
	est := e.estimateOutput(request)
	if request.RequestID != "" {
		e.cache.Add(request.RequestID, est)
	}
	return est
}

func (e *SketchTokenEstimator) estimateOutput(request *fwksched.InferenceRequest) int64 {
	if magnitude, ok := e.sketch.Predict(request.Body); ok {
		return e.clampOutput(magnitude, request.Body.MaxOutputTokens)
	}
	// Cold key: defer to the static signal-rule estimate (already clamped).
	return e.SimpleTokenEstimator.EstimateOutputFromRequest(request)
}

// Observe folds the observed output length into the sketch at end-of-stream and
// drops the request's cached prediction.
func (e *SketchTokenEstimator) Observe(request *fwksched.InferenceRequest, completionTokens int64) {
	if request == nil || request.Body == nil {
		return
	}
	e.sketch.Observe(request.Body, completionTokens)
	if request.RequestID != "" {
		e.cache.Remove(request.RequestID)
	}
}
