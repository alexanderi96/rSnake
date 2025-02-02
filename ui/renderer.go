package ui

import (
	"fmt"
	"snake-game/game"
	"snake-game/game/entity"
	"snake-game/game/types"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const (
	NumAgents = 4 // Moved from game package
)

const (
	maxScores     = 200 // Maximum number of scores to show in graph
	borderPadding = 10  // Reduced padding around game area
)

type Renderer struct {
	cellSize        int32
	screenWidth     int32
	screenHeight    int32
	graphHeight     int32
	graphWidth      int32
	gameWidth       int32
	gameHeight      int32
	statsPanel      int32
	totalGridWidth  int32
	totalGridHeight int32
	offsetX         int32
	offsetY         int32
}

func NewRenderer() *Renderer {
	r := &Renderer{}
	r.UpdateDimensions()
	return r
}

func (r *Renderer) UpdateDimensions() {
	// Get window dimensions
	r.screenWidth = int32(rl.GetScreenWidth())
	r.screenHeight = int32(rl.GetScreenHeight())

	// Calculate stats panel width as 25% of screen width for better readability
	r.statsPanel = r.screenWidth / 4

	// Calculate game area dimensions (excluding stats panel)
	r.gameWidth = r.screenWidth - r.statsPanel
	r.gameHeight = r.screenHeight * 3 / 4 // Leave bottom quarter for graph

	// Update graph dimensions
	r.graphWidth = r.gameWidth - 20       // Full width minus padding
	r.graphHeight = r.screenHeight/4 - 20 // Quarter height minus padding
}

func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

func (r *Renderer) Draw(g *game.Game) {
	r.UpdateDimensions()
	rl.BeginDrawing()
	rl.ClearBackground(rl.Black)

	// Calculate dynamic sizes
	fontSize := min(int32(r.screenHeight/45), r.statsPanel/15)   // Smaller font
	lineHeight := min(int32(r.screenHeight/35), r.statsPanel/12) // Adjusted line height

	// Calculate available space for single grid after border padding
	availableWidth := r.gameWidth - (borderPadding * 2)
	availableHeight := r.gameHeight - (borderPadding * 2)

	// Calculate cell size based on available space and grid dimensions
	cellW := availableWidth / int32(g.Grid.Width)
	cellH := availableHeight / int32(g.Grid.Height)
	r.cellSize = min(cellW, cellH)

	// Calculate total grid dimensions
	r.totalGridWidth = r.cellSize * int32(g.Grid.Width)
	r.totalGridHeight = r.cellSize * int32(g.Grid.Height)

	// Calculate offset to center the single grid
	r.offsetX = borderPadding
	r.offsetY = (r.gameHeight - r.totalGridHeight) / 2

	// Draw single grid background
	rl.DrawRectangle(
		r.offsetX-1,
		r.offsetY-1,
		r.totalGridWidth+2,
		r.totalGridHeight+2,
		rl.DarkGray)

	// Draw grid lines
	for x := 0; x < g.Grid.Width; x++ {
		for y := 0; y < g.Grid.Height; y++ {
			rl.DrawRectangleLines(
				r.offsetX+int32(x*int(r.cellSize)),
				r.offsetY+int32(y*int(r.cellSize)),
				r.cellSize, r.cellSize, rl.Gray)
		}
	}

	// Get snakes once to ensure consistency
	snakes := g.GetSnakes()

	// Draw all snakes in the same grid
	for i, snake := range snakes {
		// Draw snake if not dead
		snake.Mutex.RLock()
		isDead := snake.Dead
		score := snake.Score
		snakeColor := snake.Color
		if !isDead {
			// Copy values while holding lock to minimize lock time
			body := make([]types.Point, len(snake.Body))
			copy(body, snake.Body)
			direction := snake.Direction
			snake.Mutex.RUnlock()

			// Draw snake body
			for j, p := range body {
				color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}
				if j == 0 { // Tail
					color = rl.White
				} else if j == len(body)-1 { // Head
					color = rl.Color{
						R: uint8(float32(snakeColor.R) * 1.3),
						G: uint8(float32(snakeColor.G) * 1.3),
						B: uint8(float32(snakeColor.B) * 1.3),
						A: 255,
					}
					// Draw direction indicator
					headX := r.offsetX + int32(p.X*int(r.cellSize))
					headY := r.offsetY + int32(p.Y*int(r.cellSize))
					halfCell := r.cellSize / 2
					if direction.X > 0 { // Right
						rl.DrawTriangle(
							rl.Vector2{X: float32(headX + r.cellSize), Y: float32(headY + halfCell)},
							rl.Vector2{X: float32(headX + halfCell), Y: float32(headY)},
							rl.Vector2{X: float32(headX + halfCell), Y: float32(headY + r.cellSize)},
							rl.Yellow)
					} else if direction.X < 0 { // Left
						rl.DrawTriangle(
							rl.Vector2{X: float32(headX), Y: float32(headY + halfCell)},
							rl.Vector2{X: float32(headX + halfCell), Y: float32(headY)},
							rl.Vector2{X: float32(headX + halfCell), Y: float32(headY + r.cellSize)},
							rl.Yellow)
					} else if direction.Y > 0 { // Down
						rl.DrawTriangle(
							rl.Vector2{X: float32(headX + halfCell), Y: float32(headY + r.cellSize)},
							rl.Vector2{X: float32(headX), Y: float32(headY + halfCell)},
							rl.Vector2{X: float32(headX + r.cellSize), Y: float32(headY + halfCell)},
							rl.Yellow)
					} else { // Up
						rl.DrawTriangle(
							rl.Vector2{X: float32(headX + halfCell), Y: float32(headY)},
							rl.Vector2{X: float32(headX), Y: float32(headY + halfCell)},
							rl.Vector2{X: float32(headX + r.cellSize), Y: float32(headY + halfCell)},
							rl.Yellow)
					}
				}
				rl.DrawRectangle(
					r.offsetX+int32(p.X*int(r.cellSize)),
					r.offsetY+int32(p.Y*int(r.cellSize)),
					r.cellSize, r.cellSize, color)
			}
		} else {
			snake.Mutex.RUnlock()
		}

		// Draw agent label and score
		label := fmt.Sprintf("Agent %c: %d", rune('A'+i), score)
		labelX := r.offsetX + 5
		labelY := r.offsetY - fontSize - 5 - (int32(i) * (fontSize + 5)) // Move labels above grid
		rl.DrawText(label, labelX, labelY, fontSize, rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255})
	}

	// Draw shared food
	for _, food := range g.GetStateManager().GetFoodList() {
		rl.DrawRectangle(
			r.offsetX+int32(food.X*int(r.cellSize)),
			r.offsetY+int32(food.Y*int(r.cellSize)),
			r.cellSize, r.cellSize, rl.Red)
	}

	r.drawStatsPanel(g, fontSize, lineHeight)
	r.drawPerformanceGraph(g, fontSize, snakes)
	rl.EndDrawing()
}

func (r *Renderer) drawStatsPanel(g *game.Game, fontSize, lineHeight int32) {
	statsX := r.gameWidth + 10 // Increased gap from game area
	statsY := int32(10)

	// Get snakes once to ensure consistency
	snakes := g.GetSnakes()

	// Draw stats background
	rl.DrawRectangle(statsX-5, 0, r.statsPanel+5, r.screenHeight, rl.DarkGray)

	// Draw training time and total games
	stats, _ := g.GetStats()
	sessionDuration := time.Since(stats.SessionStartTime)
	sessionHours := int(sessionDuration.Hours())
	sessionMinutes := int(sessionDuration.Minutes()) % 60
	sessionSeconds := int(sessionDuration.Seconds()) % 60

	gameDuration := time.Since(stats.GameStartTime)
	gameHours := int(gameDuration.Hours())
	gameMinutes := int(gameDuration.Minutes()) % 60
	gameSeconds := int(gameDuration.Seconds()) % 60

	rl.DrawText("Training Stats:", statsX, statsY, fontSize, rl.White)
	statsY += lineHeight
	rl.DrawText(fmt.Sprintf("Session Time: %02d:%02d:%02d", sessionHours, sessionMinutes, sessionSeconds), statsX+5, statsY, fontSize, rl.White)
	statsY += lineHeight
	rl.DrawText(fmt.Sprintf("Current Game: %02d:%02d:%02d", gameHours, gameMinutes, gameSeconds), statsX+5, statsY, fontSize, rl.White)
	statsY += lineHeight
	rl.DrawText(fmt.Sprintf("Total Rounds: %d", stats.RoundsPlayed), statsX+5, statsY, fontSize, rl.White)
	statsY += lineHeight
	rl.DrawText(fmt.Sprintf("Current Round: %d", stats.CurrentRound), statsX+5, statsY, fontSize, rl.White)
	statsY += lineHeight * 2

	// Find global best score and agent
	globalBestScore := 0
	globalBestAgent := 0
	for i, snake := range snakes {
		snake.Mutex.RLock()
		if snake.AllTimeHigh > globalBestScore {
			globalBestScore = snake.AllTimeHigh
			globalBestAgent = i
		}
		snake.Mutex.RUnlock()
	}

	// Draw global best score
	rl.DrawText("Global Best:", statsX, statsY, fontSize, rl.White)
	statsY += lineHeight
	rl.DrawText(fmt.Sprintf("Agent %c: %d", rune('A'+globalBestAgent), globalBestScore),
		statsX+5, statsY, fontSize, rl.White)
	statsY += lineHeight * 2

	// Draw detailed stats for each agent
	rl.DrawText("Agent Stats:", statsX, statsY, fontSize, rl.White)
	statsY += lineHeight
	for i, snake := range snakes {
		snake.Mutex.RLock()
		sessionHigh := snake.SessionHigh
		allTimeHigh := snake.AllTimeHigh
		avgScore := snake.AverageScore
		gamesPlayed := snake.AI.GamesPlayed
		snakeColor := snake.Color
		snake.Mutex.RUnlock()

		color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}
		rl.DrawText(fmt.Sprintf("Agent %c:", rune('A'+i)), statsX+5, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("Session High: %d", sessionHigh), statsX+10, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("All-Time High: %d", allTimeHigh), statsX+10, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("Avg Score: %.2f", avgScore), statsX+10, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("Games: %d", gamesPlayed), statsX+10, statsY, fontSize, color)
		statsY += lineHeight * 2
	}
}

func (r *Renderer) drawPerformanceGraph(g *game.Game, fontSize int32, snakes []*entity.Snake) {
	graphX := int32(10)
	graphY := r.gameHeight + 10

	// Draw graph border with thicker lines
	for i := int32(0); i < 2; i++ {
		rl.DrawRectangleLines(graphX-i, graphY-i, r.graphWidth+i*2, r.graphHeight+i*2, rl.White)
	}

	rl.DrawText("Performance History", graphX, graphY-fontSize-5, fontSize, rl.White)

	// Find max score for scaling
	maxScore := 1
	for _, snake := range snakes {
		snake.Mutex.RLock()
		for _, score := range snake.Scores {
			if score > maxScore {
				maxScore = score
			}
		}
		snake.Mutex.RUnlock()
	}

	// Draw scores for each agent
	for _, snake := range snakes {
		snake.Mutex.RLock()
		scores := make([]int, len(snake.Scores))
		copy(scores, snake.Scores)
		avgScore := snake.AverageScore
		snakeColor := snake.Color
		snake.Mutex.RUnlock()

		color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}

		if len(scores) > 1 {
			// Draw points and connect them with thicker lines
			for j := 1; j < len(scores); j++ {
				x1 := graphX + int32(float32(r.graphWidth)*float32(j-1)/float32(maxScores))
				y1 := graphY + r.graphHeight - int32(float32(r.graphHeight)*float32(scores[j-1])/float32(maxScore))
				x2 := graphX + int32(float32(r.graphWidth)*float32(j)/float32(maxScores))
				y2 := graphY + r.graphHeight - int32(float32(r.graphHeight)*float32(scores[j])/float32(maxScore))

				// Draw thicker lines by drawing multiple offset lines
				for offset := int32(-1); offset <= 1; offset++ {
					rl.DrawLine(x1, y1+offset, x2, y2+offset, color)
				}
			}

			// Draw average score line (thicker dashed)
			avgY := graphY + r.graphHeight - int32(float32(r.graphHeight)*float32(avgScore)/float32(maxScore))
			for x := graphX; x < graphX+int32(r.graphWidth); x += 7 {
				for offset := int32(-1); offset <= 1; offset++ {
					rl.DrawLine(x, avgY+offset, x+3, avgY+offset, color)
				}
			}
		}
	}

	// Draw game over text for each dead snake
	for i, snake := range snakes {
		snake.Mutex.RLock()
		isGameOver := snake.GameOver
		snakeColor := snake.Color
		snake.Mutex.RUnlock()

		if isGameOver {
			color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}
			gameOverText := fmt.Sprintf("Agent %c: Game Over! (Restarting...)", rune('A'+i))
			textWidth := rl.MeasureText(gameOverText, fontSize)
			rl.DrawText(gameOverText,
				r.offsetX+(r.totalGridWidth-int32(textWidth))/2,
				r.offsetY+r.totalGridHeight/2+(int32(i)*fontSize),
				fontSize, color)
		}
	}
}
