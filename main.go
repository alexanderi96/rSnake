package main

import (
	"encoding/json"
	"flag"
	"math/rand"
	"os"
	"path/filepath"
	"snake-game/ai"
	"snake-game/game"
	"snake-game/ui"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	speed := flag.Int("speed", 10, "Game speed in milliseconds (lower = faster)")
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
	g := game.NewGame(width, height)

	renderer := ui.NewRenderer()
	lastUpdate := time.Now()
	updateInterval := time.Duration(*speed) * time.Millisecond

	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(rl.KeyQ) {
			// Save all Q-tables and game stats
			g.Stats.EndTime = time.Now()
			statsFile := filepath.Join("data", "games", g.UUID, "stats.json")
			statsData, _ := json.MarshalIndent(g.Stats, "", "  ")
			os.WriteFile(statsFile, statsData, 0644)

			for _, snake := range g.Snakes {
				filename := ai.GetQTableFilename(g.UUID, snake.AI.UUID)
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

			// Check for snake collisions and breeding
			for i := 0; i < len(g.Snakes); i++ {
				for j := i + 1; j < len(g.Snakes); j++ {
					snake1 := g.Snakes[i]
					snake2 := g.Snakes[j]

					// Check if snakes are face to face
					head1 := snake1.Body[len(snake1.Body)-1]
					head2 := snake2.Body[len(snake2.Body)-1]

					// Check if heads are adjacent
					if abs(head1.X-head2.X)+abs(head1.Y-head2.Y) == 1 {
						// Create new agent from parents
						childAI := ai.Breed(snake1.AI, snake2.AI)

						// Add new snake at a random position
						startPos := [2]int{
							rand.Intn(g.Grid.Width),
							rand.Intn(g.Grid.Height),
						}
						newSnake := game.NewSnake(startPos, childAI, g)
						g.Snakes = append(g.Snakes, newSnake)

						// Update game stats
						g.Stats.AgentStats = append(g.Stats.AgentStats, game.AgentStats{
							UUID:         childAI.UUID,
							Score:        0,
							AverageScore: 0,
							TotalReward:  0,
							GamesPlayed:  0,
						})
					}
				}
			}

			// Save Q-tables periodically (less frequently and synchronously)
			for _, snake := range g.Snakes {
				if snake.GameOver && snake.AI.GamesPlayed%10 == 0 {
					filename := ai.GetQTableFilename(g.UUID, snake.AI.UUID)
					snake.AI.SaveQTable(filename)
				}
			}
		}

		renderer.Draw(g)
	}

	// Save final game state
	g.Stats.EndTime = time.Now()
	statsFile := filepath.Join("data", "games", g.UUID, "stats.json")
	statsData, _ := json.MarshalIndent(g.Stats, "", "  ")
	os.WriteFile(statsFile, statsData, 0644)

	// Save Q-tables sequentially when closing window to prevent I/O contention
	for _, snake := range g.Snakes {
		filename := ai.GetQTableFilename(g.UUID, snake.AI.UUID)
		snake.AI.SaveQTable(filename)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
