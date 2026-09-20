/*
Copyright 2026 The Kubernetes Authors.

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

package parserutil

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

var ErrTrailingData = errors.New("unexpected trailing data after JSON value")

// ParseRenderRequest parses a render-path request body, preserving the original
// bytes as RawBody for byte-exact forwarding while also providing a PayloadMap
// for any routing edits that may follow (e.g. model-name rewrite).
func ParseRenderRequest(data []byte) (*fwkrh.ParseResult, error) {
	payload, err := UnmarshalEnvelope(data, "prompt", "system")
	if err != nil {
		return nil, err
	}
	model, _ := payload["model"].(string)
	return &fwkrh.ParseResult{
		Body:                   &fwkrh.InferenceRequestBody{Payload: fwkrh.PayloadMap(payload), RawBody: data, Model: model, RenderRequest: true},
		SkipResponseProcessing: true,
	}, nil
}

// UnmarshalEnvelope decodes a JSON object keeping nested objects and arrays as
// json.RawMessage to preserve byte-level representation (key order, number
// formatting). Fields listed in rawFields are also kept raw regardless of type.
// Top-level scalars are decoded with UseNumber so large integers are not rounded.
func UnmarshalEnvelope(data []byte, rawFields ...string) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		if fallbackErr := Unmarshal(data, &fields); fallbackErr != nil {
			return nil, fallbackErr
		}
	}
	result := make(map[string]any, len(fields))
	for key, raw := range fields {
		if slices.Contains(rawFields, key) || (len(raw) > 0 && (raw[0] == '{' || raw[0] == '[')) {
			result[key] = raw
			continue
		}
		var value any
		var err error
		if len(raw) > 0 && raw[0] == '"' {
			err = json.Unmarshal(raw, &value)
		} else {
			err = Unmarshal(raw, &value)
		}
		if err != nil {
			return nil, err
		}
		result[key] = value
	}
	return result, nil
}

// Unmarshal decodes one JSON value, preserves numbers as json.Number, and rejects trailing data.
func Unmarshal(data []byte, v any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(v); err != nil {
		return err
	}
	switch err := decoder.Decode(&struct{}{}); {
	case errors.Is(err, io.EOF):
		return nil
	case err == nil:
		return ErrTrailingData
	default:
		return fmt.Errorf("%w: %v", ErrTrailingData, err)
	}
}

// UnmarshalMapWithRawField decodes a JSON object while preserving one field's
// original JSON representation. All other numbers are preserved as json.Number.
func UnmarshalMapWithRawField(data []byte, rawField string) (map[string]any, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		var fallback map[string]json.RawMessage
		if fallbackErr := Unmarshal(data, &fallback); fallbackErr != nil {
			return nil, fallbackErr
		}
		return nil, err
	}

	result := make(map[string]any, len(fields))
	for key, raw := range fields {
		if key == rawField {
			result[key] = raw
			continue
		}

		var value any
		if err := Unmarshal(raw, &value); err != nil {
			return nil, err
		}
		result[key] = value
	}
	return result, nil
}
