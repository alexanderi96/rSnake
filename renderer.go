package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const borderPadding = 10 // Padding around game area

type Renderer struct {
	cellSize        int32
	screenWidth     int32
	screenHeight    int32
	totalGridWidth  int32
	totalGridHeight int32
	offsetX         int32
	offsetY         int32
	stats           *GameStats
}

func NewRenderer() *Renderer {
	r := &Renderer{
		stats: NewGameStats(),
	}
	r.UpdateDimensions()
	return r
}

func (r *Renderer) UpdateDimensions() {
	r.screenWidth = int32(rl.GetScreenWidth())
	r.screenHeight = int32(rl.GetScreenHeight())
}

func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

func (r *Renderer) Draw(g *Game) {
	r.UpdateDimensions()
	rl.BeginDrawing()
	rl.ClearBackground(rl.Black)

	fontSize := int32(r.screenHeight / 45) // Dynamic font size

	// Calculate graph height
	graphHeight := int32(150)

	// Calculate available space for grid after border padding and graph
	availableWidth := r.screenWidth - (borderPadding * 2)
	availableHeight := r.screenHeight - (borderPadding * 3) - graphHeight // Extra padding for graph separation

	// Calculate cell size based on available space and grid dimensions
	cellW := availableWidth / int32(g.Grid.Width)
	cellH := availableHeight / int32(g.Grid.Height)
	r.cellSize = min(cellW, cellH)

	// Calculate total grid dimensions
	r.totalGridWidth = r.cellSize * int32(g.Grid.Width)
	r.totalGridHeight = r.cellSize * int32(g.Grid.Height)

	// Position grid at the top with padding
	r.offsetX = borderPadding
	r.offsetY = borderPadding

	// Draw grid background
	rl.DrawRectangle(r.offsetX-1, r.offsetY-1, r.totalGridWidth+2, r.totalGridHeight+2, rl.DarkGray)

	// Draw snake
	snake := g.GetSnake()
	snake.Mutex.RLock()
	isDead := snake.Dead
	score := snake.Score
	snakeColor := snake.Color
	if !isDead {
		// Copy values while holding lock to minimize lock time
		body := make([]Point, len(snake.Body))
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

	stats := r.stats.GetStats()
	if len(stats.Games) == 0 {
		return
	}

	// Draw score
	yOffset := r.offsetY + r.totalGridHeight + borderPadding
	xOffset := r.offsetX + 10
	baseLabelOffset := r.totalGridWidth / 8
	scoreLabel := fmt.Sprintf("Score: %d", score)
	rl.DrawText(scoreLabel, xOffset, yOffset, fontSize, rl.White)

	xOffset += baseLabelOffset
	// Draw game count and scores
	rl.DrawText(fmt.Sprintf("Games: %d", r.stats.GetGamesPlayed()),
		xOffset,
		yOffset,
		fontSize, rl.White)

	xOffset += baseLabelOffset

	avgScore := r.stats.GetAverageScore()
	rl.DrawText(fmt.Sprintf("Avg Score: %.1f", avgScore),
		xOffset,
		yOffset,
		fontSize, rl.Green)

	xOffset += baseLabelOffset

	maxScore := r.stats.GetMaxScore()
	rl.DrawText(fmt.Sprintf("Max Score: %d", maxScore),
		xOffset,
		yOffset,
		fontSize, rl.Green)

	xOffset += baseLabelOffset

	// Draw duration labels

	xOffset += baseLabelOffset

	avgDuration := r.stats.GetAverageDuration()
	rl.DrawText(fmt.Sprintf("Avg Duration: %.1fs", avgDuration),
		xOffset,
		yOffset,
		fontSize, rl.Purple)

	// Draw food
	food := g.GetFood()
	rl.DrawRectangle(
		r.offsetX+int32(food.X*int(r.cellSize)),
		r.offsetY+int32(food.Y*int(r.cellSize)),
		r.cellSize, r.cellSize, rl.Red)

	// Draw statistics graph
	r.drawStatsGraph()

	rl.EndDrawing()
}

func (r *Renderer) drawStatsGraph() {
	graphHeight := int32(150)
	graphWidth := r.screenWidth - (borderPadding * 2)
	graphY := r.screenHeight - graphHeight - borderPadding

	// Draw graph background
	rl.DrawRectangle(borderPadding, graphY, graphWidth, graphHeight, rl.DarkGray)

	// Get stats data
	stats := r.stats.GetStats()
	if len(stats.Games) == 0 {
		return
	}

	// Find max values for scaling
	maxScore := stats.Games[0].Score
	maxDuration := stats.Games[0].EndTime.Sub(stats.Games[0].StartTime).Seconds()
	for _, game := range stats.Games {
		if game.Score > maxScore {
			maxScore = game.Score
		}
		duration := game.EndTime.Sub(game.StartTime).Seconds()
		if duration > maxDuration {
			maxDuration = duration
		}
	}

	// Draw graph lines
	numPoints := len(stats.Games)
	if numPoints > 1 {
		pointSpacing := float32(graphWidth) / float32(numPoints-1)
		scaleY := float32(graphHeight-40) / float32(maxScore) // Leave more padding for labels
		durationScaleY := float32(graphHeight-40) / float32(maxDuration)

		// Draw score line (green)
		for i := 0; i < numPoints; i++ {
			x := float32(borderPadding) + float32(i)*pointSpacing
			y := float32(graphY+graphHeight) - float32(stats.Games[i].Score)*scaleY

			// Draw point
			rl.DrawCircle(int32(x), int32(y), 3, rl.Green)

			// Draw line to next point
			if i < numPoints-1 {
				nextX := float32(borderPadding) + float32(i+1)*pointSpacing
				nextY := float32(graphY+graphHeight) - float32(stats.Games[i+1].Score)*scaleY
				rl.DrawLine(int32(x), int32(y), int32(nextX), int32(nextY), rl.Green)
			}
		}

		// Draw duration line (purple)
		for i := 0; i < numPoints; i++ {
			x := float32(borderPadding) + float32(i)*pointSpacing
			duration := float32(stats.Games[i].EndTime.Sub(stats.Games[i].StartTime).Seconds())
			y := float32(graphY+graphHeight) - duration*durationScaleY

			// Draw point
			rl.DrawCircle(int32(x), int32(y), 3, rl.Purple)

			// Draw line to next point
			if i < numPoints-1 {
				nextX := float32(borderPadding) + float32(i+1)*pointSpacing
				nextDuration := float32(stats.Games[i+1].EndTime.Sub(stats.Games[i+1].StartTime).Seconds())
				nextY := float32(graphY+graphHeight) - nextDuration*durationScaleY
				rl.DrawLine(int32(x), int32(y), int32(nextX), int32(nextY), rl.Purple)
			}
		}

	}
}
