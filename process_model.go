package decalgo

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
)

// ProcessModel connects to a persistent line-delimited JSON model process.
// The bundled python/hf_model.py implements this protocol with Transformers/PyTorch.
type ProcessModel struct {
	cmd         *exec.Cmd
	in          io.WriteCloser
	out         *bufio.Reader
	fingerprint string
	mu          sync.Mutex
}

type modelRequest struct {
	Op            string `json:"op"`
	Text          string `json:"text,omitempty"`
	Tokens        []int  `json:"tokens,omitempty"`
	VisibleTokens []int  `json:"visible_tokens"`
	TopN          int    `json:"top_n,omitempty"`
}

type modelResponse struct {
	OK          bool             `json:"ok"`
	Error       string           `json:"error,omitempty"`
	Fingerprint string           `json:"fingerprint,omitempty"`
	Text        string           `json:"text,omitempty"`
	Tokens      []int            `json:"tokens,omitempty"`
	Candidates  []TokenCandidate `json:"candidates,omitempty"`
}

func NewProcessModel(ctx context.Context, command string, args ...string) (*ProcessModel, error) {
	cmd := exec.CommandContext(ctx, command, args...)
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr bytesBuffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	m := &ProcessModel{cmd: cmd, in: in, out: bufio.NewReader(out)}
	response, err := m.call(modelRequest{Op: "info"})
	if err != nil {
		_ = m.Close()
		if stderr.String() != "" {
			return nil, fmt.Errorf("start model process: %w: %s", err, stderr.String())
		}
		return nil, fmt.Errorf("start model process: %w", err)
	}
	if response.Fingerprint == "" {
		_ = m.Close()
		return nil, fmt.Errorf("model process returned an empty fingerprint")
	}
	m.fingerprint = response.Fingerprint
	return m, nil
}

// A small private buffer avoids exposing an otherwise unnecessary bytes import
// in the public model API.
type bytesBuffer struct{ b []byte }

func (b *bytesBuffer) Write(p []byte) (int, error) { b.b = append(b.b, p...); return len(p), nil }
func (b *bytesBuffer) String() string              { return string(b.b) }

func (m *ProcessModel) Fingerprint() string { return m.fingerprint }

func (m *ProcessModel) Tokenize(ctx context.Context, text string) ([]int, error) {
	r, err := m.callContext(ctx, modelRequest{Op: "tokenize", Text: text})
	return r.Tokens, err
}

func (m *ProcessModel) Detokenize(ctx context.Context, tokens []int) (string, error) {
	r, err := m.callContext(ctx, modelRequest{Op: "detokenize", Tokens: tokens})
	return r.Text, err
}

func (m *ProcessModel) Next(ctx context.Context, tokens []int, topN int) ([]TokenCandidate, error) {
	r, err := m.callContext(ctx, modelRequest{Op: "next", Tokens: tokens, TopN: topN})
	return r.Candidates, err
}

// NextCopySafe excludes token transitions that a tokenizer would merge when
// the generated carrier is copied as plain text.
func (m *ProcessModel) NextCopySafe(ctx context.Context, tokens, visibleTokens []int, topN int) ([]TokenCandidate, error) {
	r, err := m.callContext(ctx, modelRequest{Op: "next", Tokens: tokens, VisibleTokens: visibleTokens, TopN: topN})
	return r.Candidates, err
}

func (m *ProcessModel) callContext(ctx context.Context, req modelRequest) (modelResponse, error) {
	select {
	case <-ctx.Done():
		return modelResponse{}, ctx.Err()
	default:
	}
	return m.call(req)
}

func (m *ProcessModel) call(req modelRequest) (modelResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	b, err := json.Marshal(req)
	if err != nil {
		return modelResponse{}, err
	}
	b = append(b, '\n')
	if _, err := m.in.Write(b); err != nil {
		return modelResponse{}, err
	}
	line, err := m.out.ReadBytes('\n')
	if err != nil {
		return modelResponse{}, err
	}
	var response modelResponse
	if err := json.Unmarshal(line, &response); err != nil {
		return modelResponse{}, fmt.Errorf("invalid model response: %w", err)
	}
	if !response.OK {
		return modelResponse{}, fmt.Errorf("model process: %s", response.Error)
	}
	return response, nil
}

func (m *ProcessModel) Close() error {
	if m.in != nil {
		_ = m.in.Close()
	}
	if m.cmd != nil && m.cmd.Process != nil {
		_ = m.cmd.Process.Kill()
	}
	if m.cmd != nil {
		return m.cmd.Wait()
	}
	return nil
}
