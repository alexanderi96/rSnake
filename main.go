package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

var shouldBeProfiled = true

func main() {
	if shouldBeProfiled {
		f, err := os.Create("cpu.pprof")
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	speed := flag.Int("speed", 1, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	rl.InitWindow(600, 600, "Snake Game")
	rl.SetWindowMinSize(600, 600)
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(999)

	// Calculate initial grid dimensions
	width := (rl.GetScreenWidth() - 20) / 20
	height := (rl.GetScreenHeight() - 190) / 20
	game := NewGame(width, height)
	agent := NewSnakeAgent(game)
	renderer := NewRenderer()
	ticker := time.NewTicker(time.Duration(*speed) * time.Millisecond)
	defer ticker.Stop()

	var gameStartTime time.Time

	// Main game loop
	for !rl.WindowShouldClose() {
		snake := game.GetSnake()

		// Start timing when snake is alive
		if !snake.Dead && gameStartTime.IsZero() {
			gameStartTime = time.Now()
		}

		// Check if snake is dead and needs to be reset
		if snake.Dead {
			// Save stats before reset
			renderer.stats.AddGame(snake.Score, gameStartTime, time.Now())
			gameStartTime = time.Time{} // Reset start time
			// Reset agent with new game
			agent.Reset()
			game = agent.game // Update our game reference
		}

		// Handle window resize
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			width := (rl.GetScreenWidth() - 20) / 20
			height := (rl.GetScreenHeight() - 190) / 20
			game.Grid.Width = width
			game.Grid.Height = height
		}

		// Update game state at fixed interval
		select {
		case <-ticker.C:
			agent.Update() // Let agent make decision and update game
		default:
		}

		// Handle quit
		if rl.IsKeyPressed(rl.KeyQ) || rl.IsKeyPressed(rl.KeyEscape) {
			if err := agent.SaveQTable(); err != nil {
				fmt.Printf("Error saving QTable: %v\n", err)
			}
			if err := renderer.stats.saveToFile(); err != nil {
				fmt.Printf("Error saving stats: %v\n", err)
			}
			break
		}

		// Render
		renderer.Draw(game)
	}
}
