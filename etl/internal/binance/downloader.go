package binance

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

type Downloader struct {
	OutputDir string
	Client    *http.Client
}

func NewDownloader(outputDir string) *Downloader {
	return &Downloader{
		OutputDir: outputDir,
		Client:    &http.Client{},
	}
}

// DownloadFile downloads a file from url and saves it to OutputDir/filename.
func (d *Downloader) DownloadFile(url, filename string) error {
	// Create output dir if not exists
	if err := os.MkdirAll(d.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}

	destPath := filepath.Join(d.OutputDir, filename)

	// Check if already exists (skip if so? or overwrite? usually skip or overwrite based on flag. For now overwrite)
	// For simplicity, just download.

	resp, err := d.Client.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save file: %w", err)
	}

	return nil
}
