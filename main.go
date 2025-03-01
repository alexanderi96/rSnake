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

	speed := flag.Int("speed", 1000, "Game speed in milliseconds (lower = faster)")
	flag.Parse()

	rl.InitWindow(900, 600, "Snake Game")
	rl.SetWindowMinSize(600, 600)
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(999)

	// Calculate initial grid dimensions
	width := rl.GetScreenWidth() / 20
	height := rl.GetScreenHeight() / 20
	game := NewGame(width, height)
	agent := NewSnakeAgent(game)
	renderer := NewRenderer()
	renderer.stats = game.Stats // Initialize renderer with game's stats
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
			renderer.stats.AddGame(snake.Score, gameStartTime, time.Now(), agent.GetEpsilon())
			gameStartTime = time.Time{} // Reset start time

			// Reset agent with new game
			agent.Reset()
			game = agent.game           // Update our game reference
			renderer.stats = game.Stats // Ensure renderer uses the same stats instance
		}

		// Handle window resize
		if rl.IsWindowResized() {
			renderer.UpdateDimensions()
			newWidth := rl.GetScreenWidth() / 20
			newHeight := rl.GetScreenHeight() / 20

			// Verifica e correggi la posizione del serpente e del cibo prima di ridimensionare
			snake := game.GetSnake()
			snake.Mutex.Lock()
			for i := 0; i < len(snake.Body); i++ {
				if snake.Body[i].X >= newWidth {
					snake.Body[i].X = newWidth - 1
				}
				if snake.Body[i].Y >= newHeight {
					snake.Body[i].Y = newHeight - 1
				}
			}
			snake.Mutex.Unlock()

			// Correggi la posizione del cibo se necessario
			if game.food.X >= newWidth {
				game.food.X = newWidth - 1
			}
			if game.food.Y >= newHeight {
				game.food.Y = newHeight - 1
			}

			// Aggiorna le dimensioni della griglia
			fmt.Printf("\nRidimensionamento griglia: %dx%d -> %dx%d\n", game.Grid.Width, game.Grid.Height, newWidth, newHeight)
			game.Grid.Width = newWidth
			game.Grid.Height = newHeight

			// Verifica posizioni dopo ridimensionamento
			snake = game.GetSnake()
			snake.Mutex.RLock()
			head := snake.GetHead()
			snake.Mutex.RUnlock()
			fmt.Printf("Posizione serpente dopo ridimensionamento: (%d,%d)\n", head.X, head.Y)
			food := game.GetFood()
			fmt.Printf("Posizione cibo dopo ridimensionamento: (%d,%d)\n", food.X, food.Y)
		}

		// Update game state at fixed interval
		select {
		case <-ticker.C:
			agent.Update() // Let agent make decision and update game
		default:
		}

		// Handle quit
		if rl.IsKeyPressed(rl.KeyQ) || rl.IsKeyPressed(rl.KeyEscape) {
			if err := agent.SaveWeights(); err != nil {
				fmt.Printf("Error saving weights: %v\n", err)
			}
			if err := renderer.stats.SaveToFile(); err != nil {
				fmt.Printf("Error saving stats: %v\n", err)
			}
			break
		}

		// Render
		renderer.Draw(game)
	}
}
