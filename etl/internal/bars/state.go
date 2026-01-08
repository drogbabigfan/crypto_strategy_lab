package bars

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

const stateFileName = "generator_state.json"

// ErrCorruptedState is returned when state validation fails.
var ErrCorruptedState = fmt.Errorf("corrupted state")

// Validate checks if the generator state is valid.
// Returns an error if the state is corrupted and should be discarded.
func (s *GeneratorState) Validate() error {
	// DailyVolumeCount must be within valid range [0, WarmupDays]
	if s.DailyVolumeCount < 0 || s.DailyVolumeCount > WarmupDays {
		return fmt.Errorf("%w: DailyVolumeCount out of range: %d", ErrCorruptedState, s.DailyVolumeCount)
	}

	// DailyVolumeHead must be within valid range [0, WarmupDays)
	if s.DailyVolumeHead < 0 || s.DailyVolumeHead >= WarmupDays {
		return fmt.Errorf("%w: DailyVolumeHead out of range: %d", ErrCorruptedState, s.DailyVolumeHead)
	}

	// CurrentThreshold must be non-negative
	if s.CurrentThreshold < 0 {
		return fmt.Errorf("%w: CurrentThreshold is negative: %f", ErrCorruptedState, s.CurrentThreshold)
	}
	if math.IsNaN(s.CurrentThreshold) || math.IsInf(s.CurrentThreshold, 0) {
		return fmt.Errorf("%w: CurrentThreshold is NaN or Inf", ErrCorruptedState)
	}

	// CurrentDayVolume must be non-negative
	if s.CurrentDayVolume < 0 {
		return fmt.Errorf("%w: CurrentDayVolume is negative: %f", ErrCorruptedState, s.CurrentDayVolume)
	}
	if math.IsNaN(s.CurrentDayVolume) || math.IsInf(s.CurrentDayVolume, 0) {
		return fmt.Errorf("%w: CurrentDayVolume is NaN or Inf", ErrCorruptedState)
	}

	// DailyVolumes must not contain negative, NaN, or Inf values
	for i := 0; i < s.DailyVolumeCount; i++ {
		v := s.DailyVolumes[i]
		if v < 0 {
			return fmt.Errorf("%w: DailyVolumes[%d] is negative: %f", ErrCorruptedState, i, v)
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("%w: DailyVolumes[%d] is NaN or Inf", ErrCorruptedState, i)
		}
	}

	// EMAVolume must be non-negative (can be 0 during warmup)
	if s.EMAVolume < 0 {
		return fmt.Errorf("%w: EMAVolume is negative: %f", ErrCorruptedState, s.EMAVolume)
	}
	if math.IsNaN(s.EMAVolume) || math.IsInf(s.EMAVolume, 0) {
		return fmt.Errorf("%w: EMAVolume is NaN or Inf", ErrCorruptedState)
	}

	// Accumulator validation
	if s.Accumulator.TickCount < 0 {
		return fmt.Errorf("%w: Accumulator.TickCount is negative: %d", ErrCorruptedState, s.Accumulator.TickCount)
	}
	if s.Accumulator.DollarValue < 0 {
		return fmt.Errorf("%w: Accumulator.DollarValue is negative: %f", ErrCorruptedState, s.Accumulator.DollarValue)
	}
	if math.IsNaN(s.Accumulator.DollarValue) || math.IsInf(s.Accumulator.DollarValue, 0) {
		return fmt.Errorf("%w: Accumulator.DollarValue is NaN or Inf", ErrCorruptedState)
	}

	// OHLC sanity check (if there are ticks, prices should be set)
	if s.Accumulator.TickCount > 0 {
		if s.Accumulator.High < s.Accumulator.Low {
			return fmt.Errorf("%w: Accumulator High < Low: %f < %f", ErrCorruptedState, s.Accumulator.High, s.Accumulator.Low)
		}
	}

	// LastProcessedTradeID must be non-negative
	if s.LastProcessedTradeID < 0 {
		return fmt.Errorf("%w: LastProcessedTradeID is negative: %d", ErrCorruptedState, s.LastProcessedTradeID)
	}

	return nil
}

// GetStateFilePath returns the state file path for a given directory.
// Since we now use symbol-specific directories, we just use a simple filename.
func GetStateFilePath(stateDir, symbol string) string {
	// Symbol parameter kept for backward compatibility but not used in filename
	// since the stateDir is already symbol-specific
	return filepath.Join(stateDir, stateFileName)
}

// SaveStateForSymbol saves the generator state to a symbol-specific JSON file.
func SaveStateForSymbol(stateDir, symbol string, state GeneratorState) error {
	statePath := GetStateFilePath(stateDir, symbol)

	// Ensure directory exists
	if err := os.MkdirAll(stateDir, 0755); err != nil {
		return fmt.Errorf("failed to create state directory: %w", err)
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	if err := os.WriteFile(statePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write state file: %w", err)
	}

	return nil
}

// LoadStateForSymbol loads the generator state from a symbol-specific JSON file.
// Returns nil state and no error if the file doesn't exist.
// If the state is corrupted (fails validation), it returns nil state with an error.
func LoadStateForSymbol(stateDir, symbol string) (*GeneratorState, error) {
	statePath := GetStateFilePath(stateDir, symbol)

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

	// Validate loaded state - corrupted state should be discarded
	if err := state.Validate(); err != nil {
		return nil, err
	}

	return &state, nil
}

// DeleteStateForSymbol removes the symbol-specific state file.
func DeleteStateForSymbol(stateDir, symbol string) error {
	statePath := GetStateFilePath(stateDir, symbol)
	err := os.Remove(statePath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete state file: %w", err)
	}
	return nil
}

// SaveState saves the generator state to a JSON file in the output directory.
// DEPRECATED: Use SaveStateForSymbol for multi-symbol support.
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

// LoadState loads the generator state from a JSON file in the output directory.
// Returns nil state and no error if the file doesn't exist.
// If the state is corrupted (fails validation), it returns nil state with an error.
// The caller should handle this by starting fresh (discarding the corrupted state).
// DEPRECATED: Use LoadStateForSymbol for multi-symbol support.
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

	// Validate loaded state - corrupted state should be discarded
	if err := state.Validate(); err != nil {
		return nil, err
	}

	return &state, nil
}

// DeleteState removes the state file.
// DEPRECATED: Use DeleteStateForSymbol for multi-symbol support.
func DeleteState(outDir string) error {
	statePath := filepath.Join(outDir, stateFileName)
	err := os.Remove(statePath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete state file: %w", err)
	}
	return nil
}

// MigrateState migrates legacy state file to symbol-specific format.
// If legacy state exists and symbol-specific doesn't, moves the state.
// Returns true if migration occurred.
func MigrateState(outDir, stateDir, symbol string) (bool, error) {
	legacyPath := filepath.Join(outDir, stateFileName)
	newPath := GetStateFilePath(stateDir, symbol)

	// Check if legacy state exists
	if _, err := os.Stat(legacyPath); os.IsNotExist(err) {
		return false, nil // No legacy state to migrate
	}

	// Check if new state already exists
	if _, err := os.Stat(newPath); err == nil {
		return false, nil // New state already exists, no migration needed
	}

	// Ensure state directory exists
	if err := os.MkdirAll(stateDir, 0755); err != nil {
		return false, fmt.Errorf("failed to create state directory: %w", err)
	}

	// Read legacy state
	data, err := os.ReadFile(legacyPath)
	if err != nil {
		return false, fmt.Errorf("failed to read legacy state: %w", err)
	}

	// Write to new location
	if err := os.WriteFile(newPath, data, 0644); err != nil {
		return false, fmt.Errorf("failed to write new state: %w", err)
	}

	// Optionally, keep legacy file as backup (don't delete)
	// Or delete it: os.Remove(legacyPath)

	return true, nil
}
