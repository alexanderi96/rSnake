package main

import (
	"flag"
	"math/rand"
	"snake-game/ai"
	"snake-game/game"
	"snake-game/game/types"
	"snake-game/ui"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	speed := flag.Int("speed", 20, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	rl.InitWindow(1280, 800, "Snake AI - Q-Learning")
	// rl.SetWindowMinSize(800, 600) // Set minimum window size
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	// Calculate initial grid dimensions based on window size
	// Use larger grid since we're displaying all snakes in one grid
	width := (rl.GetScreenWidth() - 300) / 15  // -300 for stats panel and padding
	height := (rl.GetScreenHeight() - 40) / 15 // -40 for padding
	g := game.NewGame(width, height, "")

	renderer := ui.NewRenderer()
	lastUpdate := time.Now()
	updateInterval := time.Duration(*speed) * time.Millisecond

	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(rl.KeyQ) {
			// Save all Q-tables and game stats
			g.SaveGameStats() // Use the proper SaveGameStats method

			for _, snake := range g.GetSnakes() {
				filename := ai.GetQTableFilename(g.GetUUID(), snake.AI.UUID)
				snake.AI.SaveQTable(filename)
			}
			break
		}

		// Handle window resize
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			// Update grid dimensions on resize
			width := (rl.GetScreenWidth() - 300) / 15  // -300 for stats panel and padding
			height := (rl.GetScreenHeight() - 40) / 15 // -40 for padding
			g.Grid.Width = width
			g.Grid.Height = height
		}

		// Update game state at fixed interval
		if time.Since(lastUpdate) >= updateInterval {
			g.Update()
			lastUpdate = time.Now()

			// Check if all snakes are dead to start a new game
			if g.GetStateManager().GetPopulationManager().IsAllSnakesDead() {
				// Save current game stats
				g.SaveGameStats()

				// Save Q-tables for all snakes
				for _, snake := range g.GetSnakes() {
					filename := ai.GetQTableFilename(g.GetUUID(), snake.AI.UUID)
					snake.AI.SaveQTable(filename)
				}

				// Get best snakes from current game
				bestSnakes := g.GetStateManager().GetPopulationManager().GetBestSnakes(types.NumAgents)
				previousGameID := g.GetUUID()

				// Create new game with same dimensions
				g = game.NewGame(g.Grid.Width, g.Grid.Height, previousGameID)

				// Add mutated copies of best snakes to new game
				for _, bestSnake := range bestSnakes {
					// Create new agent with mutation of best snake's Q-table
					newAgent := ai.NewQLearning(bestSnake.AI.GetQTable(), 0.01) // 1% mutation rate

					// Add new snake at a random position
					startPos := [2]int{
						rand.Intn(g.Grid.Width),
						rand.Intn(g.Grid.Height),
					}
					g.NewSnake(startPos[0], startPos[1], newAgent)
				}
			}

			// Save Q-tables periodically
			for _, snake := range g.GetSnakes() {
				if snake.GameOver && snake.AI.GamesPlayed%10 == 0 {
					filename := ai.GetQTableFilename(g.GetUUID(), snake.AI.UUID)
					snake.AI.SaveQTable(filename)
				}
			}
		}

		renderer.Draw(g)
	}

	// Save final game state
	g.SaveGameStats() // Use the proper SaveGameStats method

	// Save Q-tables sequentially when closing window to prevent I/O contention
	for _, snake := range g.GetSnakes() {
		filename := ai.GetQTableFilename(g.GetUUID(), snake.AI.UUID)
		snake.AI.SaveQTable(filename)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
