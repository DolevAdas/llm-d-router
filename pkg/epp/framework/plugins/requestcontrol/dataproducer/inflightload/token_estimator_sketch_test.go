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
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/outlenbucket"
)

// fakeSketch is a scripted sketchModel: it predicts a fixed magnitude (or abstains)
// and records predict/observe calls.
type fakeSketch struct {
	predict      int64
	ok           bool
	predictCalls int
	observed     []int64
}

func (f *fakeSketch) Predict(*fwkrh.InferenceRequestBody) (int64, bool) {
	f.predictCalls++
	return f.predict, f.ok
}

func (f *fakeSketch) Observe(_ *fwkrh.InferenceRequestBody, completionTokens int64) {
	f.observed = append(f.observed, completionTokens)
}

func newSketchEstimator(t *testing.T, s sketchModel, maxOutput *int64) *SketchTokenEstimator {
	t.Helper()
	e, err := NewSketchTokenEstimator(s, maxOutput)
	if err != nil {
		t.Fatalf("NewSketchTokenEstimator: %v", err)
	}
	return e
}

func sketchRequest(id string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{RequestID: id, Body: &fwkrh.InferenceRequestBody{}}
}

func TestSketchEstimatorUsesPrediction(t *testing.T) {
	e := newSketchEstimator(t, &fakeSketch{predict: 3200, ok: true}, nil)
	if got := e.EstimateOutputFromRequest(sketchRequest("r1")); got != 3200 {
		t.Errorf("estimate = %d, want 3200 (sketch prediction)", got)
	}
}

func TestSketchEstimatorColdFallsBackToStatic(t *testing.T) {
	// Cold sketch (ok=false) with a LONG outlen-bucket attribute falls back to the
	// static long estimate.
	e := newSketchEstimator(t, &fakeSketch{ok: false}, nil)
	req := sketchRequest("r1")
	req.PutAttribute(outlenbucket.AttributeKey, outlenbucket.Long)
	if got := e.EstimateOutputFromRequest(req); got != LongOutputTokens {
		t.Errorf("cold estimate = %d, want static LONG %d", got, LongOutputTokens)
	}
}

func TestSketchEstimatorCachesPerRequest(t *testing.T) {
	f := &fakeSketch{predict: 1500, ok: true}
	e := newSketchEstimator(t, f, nil)
	req := sketchRequest("r1")
	for i := 0; i < 5; i++ {
		e.EstimateOutputFromRequest(req)
	}
	if f.predictCalls != 1 {
		t.Errorf("predict called %d times, want 1 (cached per request)", f.predictCalls)
	}
	// A different request id is predicted independently.
	e.EstimateOutputFromRequest(sketchRequest("r2"))
	if f.predictCalls != 2 {
		t.Errorf("predict called %d times across two requests, want 2", f.predictCalls)
	}
}

func TestSketchEstimatorObserveClearsCache(t *testing.T) {
	f := &fakeSketch{predict: 1500, ok: true}
	e := newSketchEstimator(t, f, nil)
	req := sketchRequest("r1")
	e.EstimateOutputFromRequest(req)
	e.Observe(req, 1800)
	e.EstimateOutputFromRequest(req) // cache cleared -> recompute

	if len(f.observed) != 1 || f.observed[0] != 1800 {
		t.Errorf("observed = %v, want [1800]", f.observed)
	}
	if f.predictCalls != 2 {
		t.Errorf("predict called %d times, want 2 (observe cleared the cache)", f.predictCalls)
	}
}

func TestSketchEstimatorClampsPrediction(t *testing.T) {
	cap := int64(500)
	e := newSketchEstimator(t, &fakeSketch{predict: 9000, ok: true}, &cap)
	if got := e.EstimateOutputFromRequest(sketchRequest("r1")); got != 500 {
		t.Errorf("estimate = %d, want 500 (operator cap)", got)
	}
}

func TestBuildTokenEstimatorSelection(t *testing.T) {
	if _, err := buildTokenEstimator(Config{}); err != nil {
		t.Errorf("default (static) estimator: unexpected error %v", err)
	}
	if _, err := buildTokenEstimator(Config{OutputEstimator: "sketch"}); err == nil {
		t.Error("sketch estimator without modelDir should error")
	}
	if _, err := buildTokenEstimator(Config{OutputEstimator: "bogus"}); err == nil {
		t.Error("unknown estimator should error")
	}
}
