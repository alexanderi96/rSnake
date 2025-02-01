package ui

import (
	"fmt"
	"snake-game/game"
	"time"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const (
	maxScores     = 200 // Maximum number of scores to show in graph
	borderPadding = 20  // Padding around game area
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

	// Calculate stats panel width as a percentage of screen width (20%)
	r.statsPanel = r.screenWidth / 5

	// Calculate game area dimensions (excluding stats panel)
	r.gameWidth = r.screenWidth - r.statsPanel
	r.gameHeight = r.screenHeight

	// Ensure game area and stats panel cover full window
	r.statsPanel = r.screenWidth - r.gameWidth

	// Update graph dimensions
	r.graphWidth = r.screenWidth - 40  // Full width minus padding
	r.graphHeight = r.screenHeight / 4 // Graph takes up 1/4 of screen height
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
	fontSize := min(int32(r.screenHeight/40), r.statsPanel/15)   // Dynamic font size based on screen height and panel width
	lineHeight := min(int32(r.screenHeight/32), r.statsPanel/12) // Dynamic line height based on screen height and panel width

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
	r.offsetX = borderPadding + (availableWidth-r.totalGridWidth)/2
	r.offsetY = borderPadding + (availableHeight-r.totalGridHeight)/2

	// Draw single grid background
	rl.DrawRectangle(
		r.offsetX-1,
		r.offsetY-1,
		r.totalGridWidth+2,
		r.totalGridHeight+2,
		rl.Black)

	// Draw grid lines
	for x := 0; x < g.Grid.Width; x++ {
		for y := 0; y < g.Grid.Height; y++ {
			rl.DrawRectangleLines(
				r.offsetX+int32(x*int(r.cellSize)),
				r.offsetY+int32(y*int(r.cellSize)),
				r.cellSize, r.cellSize, rl.Gray)
		}
	}

	// Draw all snakes in the same grid
	for i, snake := range g.Snakes {
		// Draw snake if not dead
		snake.Mutex.RLock()
		isDead := snake.Dead
		score := snake.Score
		snakeColor := snake.Color // Get snake's custom color
		if !isDead {
			// Copy values while holding lock to minimize lock time
			body := make([]game.Point, len(snake.Body))
			copy(body, snake.Body)
			food := snake.Food
			direction := snake.Direction
			snake.Mutex.RUnlock()

			// Draw snake body
			for j, p := range body {
				color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}
				if j == 0 { // Tail
					color = rl.White // White for tail
				} else if j == len(body)-1 { // Head
					// Make head brighter to stand out
					color = rl.Color{
						R: uint8(float32(snakeColor.R) * 1.3),
						G: uint8(float32(snakeColor.G) * 1.3),
						B: uint8(float32(snakeColor.B) * 1.3),
						A: 255,
					}
					// Draw direction indicator (small triangle in movement direction)
					headX := r.offsetX + int32(p.X*int(r.cellSize))
					headY := r.offsetY + int32(p.Y*int(r.cellSize))
					halfCell := r.cellSize / 2
					// Draw triangle based on direction
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

			// Draw food for this grid
			rl.DrawRectangle(
				r.offsetX+int32(food.X*int(r.cellSize)),
				r.offsetY+int32(food.Y*int(r.cellSize)),
				r.cellSize, r.cellSize, rl.Red)
		} else {
			snake.Mutex.RUnlock()
		}

		// Draw agent label and score
		label := fmt.Sprintf("Agent %c: %d", rune('A'+i), score)
		labelX := r.offsetX + 5
		labelY := r.offsetY + 5 + (int32(i) * (fontSize + 5)) // Stack labels vertically
		rl.DrawText(label, labelX, labelY, fontSize, rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255})
	}

	r.drawStatsPanel(g, fontSize, lineHeight)
	rl.EndDrawing()
}

func (r *Renderer) drawStatsPanel(g *game.Game, fontSize, lineHeight int32) {
	statsX := r.gameWidth
	statsY := int32(10)

	// Draw stats background
	rl.DrawRectangle(statsX, 0, r.statsPanel+1, r.screenHeight, rl.DarkGray)

	// Helper function to format float with 2 decimal places
	formatFloat := func(f float64) string {
		return fmt.Sprintf("%.2f", f)
	}

	// Draw high scores for each agent
	rl.DrawText("High Scores:", statsX+10, statsY, fontSize, rl.White)
	statsY += lineHeight
	for _, snake := range g.Snakes {
		snake.Mutex.RLock()
		sessionHigh := snake.SessionHigh
		allTimeHigh := snake.AllTimeHigh
		snakeColor := snake.Color
		snake.Mutex.RUnlock()

		color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}
		rl.DrawText(fmt.Sprintf("Agent %c:", rune('A')), statsX+20, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("Session: %d", sessionHigh), statsX+30, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("All-Time: %d", allTimeHigh), statsX+30, statsY, fontSize, color)
		statsY += lineHeight
	}
	statsY += lineHeight

	// Draw agent statistics
	rl.DrawText("Agent Stats:", statsX+10, statsY, fontSize, rl.White)
	statsY += lineHeight
	for i := 0; i < game.NumAgents; i++ {
		currentSnake := g.Snakes[i]
		currentSnake.Mutex.RLock()
		avgScore := currentSnake.AverageScore
		gamesPlayed := currentSnake.AI.GamesPlayed
		snakeColor := currentSnake.Color
		currentSnake.Mutex.RUnlock()

		color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}
		rl.DrawText(fmt.Sprintf("Agent %c", rune('A'+i)), statsX+20, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("  Avg: %s", formatFloat(avgScore)), statsX+20, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("  Games: %d", gamesPlayed), statsX+20, statsY, fontSize, color)
		statsY += lineHeight + lineHeight/2
	}

	r.drawPerformanceGraph(g, statsX, statsY, fontSize)
}

func (r *Renderer) drawPerformanceGraph(g *game.Game, statsX, statsY, fontSize int32) {
	// Draw performance graph at the bottom of screen
	graphX := int32(20) // Left padding
	graphHeight := r.graphHeight
	graphY := r.screenHeight - graphHeight - fontSize*3 // More space for training info
	rl.DrawRectangleLines(graphX, graphY, r.graphWidth, graphHeight, rl.White)
	rl.DrawText("Performance", graphX, graphY-20, fontSize, rl.White)

	// Draw training time and total games
	trainingDuration := time.Since(g.StartTime)
	hours := int(trainingDuration.Hours())
	minutes := int(trainingDuration.Minutes()) % 60
	seconds := int(trainingDuration.Seconds()) % 60
	timeText := fmt.Sprintf("Training Time: %02d:%02d:%02d - Total Games: %d", hours, minutes, seconds, g.TotalGames)
	rl.DrawText(timeText, graphX, r.screenHeight-fontSize-5, fontSize, rl.White)

	// Find global max score for scaling
	maxScore := 1
	for _, snake := range g.Snakes {
		for _, score := range snake.Scores {
			if score > maxScore {
				maxScore = score
			}
		}
	}

	// Draw scores for each agent
	graphWidth := r.graphWidth
	for _, snake := range g.Snakes {
		snake.Mutex.RLock()
		scores := make([]int, len(snake.Scores))
		copy(scores, snake.Scores)
		avgScore := snake.AverageScore
		snakeColor := snake.Color
		snake.Mutex.RUnlock()

		color := rl.Color{R: snakeColor.R, G: snakeColor.G, B: snakeColor.B, A: 255}

		if len(scores) > 1 {
			// Draw points and connect them with lines
			for j := 1; j < len(scores); j++ {
				x1 := graphX + int32(float32(graphWidth)*float32(j-1)/float32(maxScores))
				y1 := graphY + graphHeight - int32(float32(graphHeight)*float32(scores[j-1])/float32(maxScore))
				x2 := graphX + int32(float32(graphWidth)*float32(j)/float32(maxScores))
				y2 := graphY + graphHeight - int32(float32(graphHeight)*float32(scores[j])/float32(maxScore))
				rl.DrawLine(x1, y1, x2, y2, color)
				rl.DrawCircle(x2, y2, 2, color)
			}

			// Draw average score line (dashed)
			avgY := graphY + graphHeight - int32(float32(graphHeight)*float32(avgScore)/float32(maxScore))
			for x := graphX; x < graphX+int32(graphWidth); x += 5 {
				rl.DrawLine(x, avgY, x+2, avgY, color)
			}
		}
	}

	// Draw game over text for each dead snake
	for i, snake := range g.Snakes {
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
