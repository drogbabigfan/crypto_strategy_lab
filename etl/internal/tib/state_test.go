package tib

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveAndLoadState(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "tib_state_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create a state with all fields populated
	originalState := GeneratorState{
		PrevPrice: 12345.67890123,
		PrevDir:   -1,
		Imbalance: 42.5,
		CurrentBar: &Bar{
			Timestamp: 1234567890,
			Open:      100.0,
			High:      105.0,
			Low:       99.0,
			Close:     103.0,
			Volume:    1500.5,
			TickCount: 250,
		},
	}

	// Save state
	if err := SaveState(tmpDir, originalState); err != nil {
		t.Fatalf("SaveState failed: %v", err)
	}

	// Verify file exists
	statePath := filepath.Join(tmpDir, stateFileName)
	if _, err := os.Stat(statePath); os.IsNotExist(err) {
		t.Fatal("State file was not created")
	}

	// Load state
	loadedState, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}
	if loadedState == nil {
		t.Fatal("LoadState returned nil")
	}

	// Verify all fields
	if loadedState.PrevPrice != originalState.PrevPrice {
		t.Errorf("PrevPrice: expected %f, got %f", originalState.PrevPrice, loadedState.PrevPrice)
	}
	if loadedState.PrevDir != originalState.PrevDir {
		t.Errorf("PrevDir: expected %d, got %d", originalState.PrevDir, loadedState.PrevDir)
	}
	if loadedState.Imbalance != originalState.Imbalance {
		t.Errorf("Imbalance: expected %f, got %f", originalState.Imbalance, loadedState.Imbalance)
	}

	// Verify CurrentBar
	if loadedState.CurrentBar == nil {
		t.Fatal("CurrentBar should not be nil")
	}
	if loadedState.CurrentBar.Open != originalState.CurrentBar.Open {
		t.Errorf("CurrentBar.Open: expected %f, got %f", originalState.CurrentBar.Open, loadedState.CurrentBar.Open)
	}
	if loadedState.CurrentBar.High != originalState.CurrentBar.High {
		t.Errorf("CurrentBar.High: expected %f, got %f", originalState.CurrentBar.High, loadedState.CurrentBar.High)
	}
	if loadedState.CurrentBar.Volume != originalState.CurrentBar.Volume {
		t.Errorf("CurrentBar.Volume: expected %f, got %f", originalState.CurrentBar.Volume, loadedState.CurrentBar.Volume)
	}
	if loadedState.CurrentBar.TickCount != originalState.CurrentBar.TickCount {
		t.Errorf("CurrentBar.TickCount: expected %d, got %d", originalState.CurrentBar.TickCount, loadedState.CurrentBar.TickCount)
	}
}

func TestLoadStateNonExistent(t *testing.T) {
	// Create temp directory (empty)
	tmpDir, err := os.MkdirTemp("", "tib_state_test_empty")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Load from empty directory should return nil without error
	state, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState should not error for non-existent file: %v", err)
	}
	if state != nil {
		t.Error("LoadState should return nil for non-existent file")
	}
}

func TestSaveStateWithNilCurrentBar(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "tib_state_test_nil_bar")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// State without CurrentBar
	state := GeneratorState{
		PrevPrice:  100.0,
		PrevDir:    1,
		Imbalance:  5.0,
		CurrentBar: nil,
	}

	// Save and load
	if err := SaveState(tmpDir, state); err != nil {
		t.Fatalf("SaveState failed: %v", err)
	}

	loaded, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}

	if loaded.CurrentBar != nil {
		t.Error("CurrentBar should be nil after load")
	}
	if loaded.PrevPrice != 100.0 {
		t.Errorf("PrevPrice: expected 100.0, got %f", loaded.PrevPrice)
	}
}

func TestStateOverwrite(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "tib_state_test_overwrite")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Save first state
	state1 := GeneratorState{PrevPrice: 100.0, PrevDir: 1, Imbalance: 1.0}
	if err := SaveState(tmpDir, state1); err != nil {
		t.Fatalf("First SaveState failed: %v", err)
	}

	// Overwrite with second state
	state2 := GeneratorState{PrevPrice: 200.0, PrevDir: -1, Imbalance: 2.0}
	if err := SaveState(tmpDir, state2); err != nil {
		t.Fatalf("Second SaveState failed: %v", err)
	}

	// Load should return the second state
	loaded, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}

	if loaded.PrevPrice != 200.0 {
		t.Errorf("Expected overwritten PrevPrice 200.0, got %f", loaded.PrevPrice)
	}
	if loaded.PrevDir != -1 {
		t.Errorf("Expected overwritten PrevDir -1, got %d", loaded.PrevDir)
	}
}
