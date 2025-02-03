package main

import (
	"flag"
	"snake-game/game"
	"snake-game/ui"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	speed := flag.Int("speed", 1, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	// rand.Seed(time.Now().UnixNano())

	rl.InitWindow(900, 800, "Snake AI - Q-Learning")
	rl.SetWindowMinSize(600, 600) // Set minimum window size
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	// Calculate initial grid dimensions
	width := (rl.GetScreenWidth() - 20) / 20    // -20 for padding
	height := (rl.GetScreenHeight() - 190) / 20 // -190 for padding and graph space
	g := game.NewGame(width, height)

	renderer := ui.NewRenderer()
	lastUpdate := time.Now()
	updateInterval := time.Duration(*speed) * time.Millisecond

	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(rl.KeyQ) {
			break
		}

		// Handle window resize
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			// Update grid dimensions on resize
			width := (rl.GetScreenWidth() - 20) / 20    // -20 for padding
			height := (rl.GetScreenHeight() - 190) / 20 // -190 for padding and graph space
			g.Grid.Width = width
			g.Grid.Height = height
		}

		// Update game state at fixed interval
		if time.Since(lastUpdate) >= updateInterval {
			g.Update()
			lastUpdate = time.Now()
		}

		renderer.Draw(g)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
