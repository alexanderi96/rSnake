package main

import (
	"flag"
	"fmt"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	speed := flag.Int("speed", 10, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	rl.InitWindow(900, 800, "Snake Game")
	rl.SetWindowMinSize(600, 600)
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	// Calculate initial grid dimensions
	width := (rl.GetScreenWidth() - 20) / 20
	height := (rl.GetScreenHeight() - 190) / 20
	game := NewGame(width, height)
	agent := NewSnakeAgent(game)
	renderer := NewRenderer()
	ticker := time.NewTicker(time.Duration(*speed) * time.Millisecond)
	defer ticker.Stop()

	// Main game loop
	for !rl.WindowShouldClose() {
		snake := game.GetSnake()

		// Check if snake is dead and needs to be reset
		if snake.Dead {
			// Save stats before reset
			renderer.stats.AddGame(snake.Score)
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
			// Save QTable before quitting
			if err := agent.SaveQTable(); err != nil {
				fmt.Printf("Error saving QTable on exit: %v\n", err)
			}
			break
		}

		// Render
		renderer.Draw(game)
	}
}
