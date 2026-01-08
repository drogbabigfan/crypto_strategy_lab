package binance

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDownloadFile(t *testing.T) {
	// Mock Server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "mock zip content")
	}))
	defer ts.Close()

	// Temporary directory for download
	tmpDir := t.TempDir()

	downloader := NewDownloader(tmpDir)

	// Test Download
	filename := "test.zip"
	url := ts.URL + "/" + filename

	err := downloader.DownloadFile(url, filename)
	if err != nil {
		t.Fatalf("DownloadFile failed: %v", err)
	}

	// Verify file exists
	destPath := filepath.Join(tmpDir, filename)
	if _, err := os.Stat(destPath); os.IsNotExist(err) {
		t.Errorf("File was not created at %s", destPath)
	}

	// Verify content
	content, err := os.ReadFile(destPath)
	if err != nil {
		t.Fatalf("Failed to read file: %v", err)
	}
	if string(content) != "mock zip content\n" {
		t.Errorf("File content mismatch. Got %s", string(content))
	}
}
