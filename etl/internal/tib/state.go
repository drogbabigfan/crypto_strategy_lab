package tib

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const stateFileName = "generator_state.json"

// SaveState saves the generator state to a JSON file in the output directory
func SaveState(outDir string, state GeneratorState) error {
	statePath := filepath.Join(outDir, stateFileName)

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	if err := os.WriteFile(statePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write state file: %w", err)
	}

	return nil
}

// LoadState loads the generator state from a JSON file in the output directory
// Returns nil state and no error if the file doesn't exist
func LoadState(outDir string) (*GeneratorState, error) {
	statePath := filepath.Join(outDir, stateFileName)

	data, err := os.ReadFile(statePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No state file, fresh start
		}
		return nil, fmt.Errorf("failed to read state file: %w", err)
	}

	var state GeneratorState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal state: %w", err)
	}

	return &state, nil
}
