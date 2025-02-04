package main

import (
	"flag"
	"fmt"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	speed := flag.Int("speed", 1, "Game speed in milliseconds (lower = faster)")
	numAgents := flag.Int("agents", 4, "Number of concurrent agents")
	resetStats := flag.Bool("R", false, "Reset stats (old stats will be backed up)")
	flag.Parse()

	rl.InitWindow(900, 800, "Multi-Agent Snake Game")

	// Create renderer (which manages stats internally)
	renderer := NewRenderer()
	if *resetStats {
		if err := renderer.stats.Reset(); err != nil {
			fmt.Printf("Error resetting stats: %v\n", err)
			return
		}
		fmt.Println("Stats reset successfully. Old stats backed up in data/backup/")
	}
	rl.SetWindowMinSize(600, 600)
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	// Calculate grid dimensions for the shared game
	baseWidth := (rl.GetScreenWidth() - 20) / 20
	baseHeight := (rl.GetScreenHeight() - 190) / 20

	// Create a single game with multiple snakes
	game := NewGame(baseWidth, baseHeight, *numAgents)

	ticker := time.NewTicker(time.Duration(*speed) * time.Millisecond)
	defer ticker.Stop()

	// Track game start times for stats
	gameStartTimes := make([]time.Time, *numAgents)

	// Main game loop
	for !rl.WindowShouldClose() {
		// Handle window resize
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			newWidth := (rl.GetScreenWidth() - 20) / 20
			newHeight := (rl.GetScreenHeight() - 190) / 20

			game.Grid.Width = newWidth
			game.Grid.Height = newHeight
		}

		// Update game state
		select {
		case <-ticker.C:
			for i, snake := range game.GetSnakes() {
				// Start timing when snake is alive
				if !snake.Dead && gameStartTimes[i].IsZero() {
					gameStartTimes[i] = time.Now()
				}

				// If snake is dead, update stats before it gets respawned
				if snake.Dead {
					renderer.stats.AddGame(snake.Score, gameStartTimes[i], time.Now())
					gameStartTimes[i] = time.Time{} // Reset start time
				}
			}

			// Update all agents (this will handle respawning dead agents)
			game.UpdateAgents()
		default:
		}

		// Handle quit
		if rl.IsKeyPressed(rl.KeyQ) || rl.IsKeyPressed(rl.KeyEscape) {
			// Save QTable before quitting (using first agent since QTable is shared)
			if len(game.agents) > 0 && game.agents[0] != nil {
				if err := game.agents[0].SaveQTable(); err != nil {
					fmt.Printf("Error saving QTable on exit: %v\n", err)
				}
			}
			break
		}

		// Pass the single game to renderer
		renderer.Draw([]*Game{game})
	}
}
