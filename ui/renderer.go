package ui

import (
	"fmt"
	"snake-game/game"
	"snake-game/game/types"

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
}

func NewRenderer() *Renderer {
	r := &Renderer{}
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

func (r *Renderer) Draw(g *game.Game) {
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

	// Get snakes once to ensure consistency
	snakes := g.GetSnakes()

	// Draw snake
	if len(snakes) > 0 {
		snake := snakes[0]
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

		// Draw score history graph
		history := g.GetStateManager().GetScoreHistory()
		if len(history) > 0 {
			statsWidth := int32(200) // Space reserved for statistics
			graphWidth := r.totalGridWidth - statsWidth
			graphX := r.offsetX + statsWidth
			graphY := r.offsetY + r.totalGridHeight + borderPadding

			// Draw scores and games count
			scoreLabel := fmt.Sprintf("Score: %d/%d\nGames: %d", score, g.GetStateManager().GetHighScore(), len(g.GetStateManager().GetScoreHistory()))
			rl.DrawText(scoreLabel, r.offsetX+10, graphY+graphHeight/2-10, fontSize, rl.White)

			// Draw graph background
			rl.DrawRectangle(graphX-1, graphY-1, graphWidth+2, graphHeight+2, rl.DarkGray)
			rl.DrawRectangle(graphX, graphY, graphWidth, graphHeight, rl.Black)

			// Find max score for scaling
			maxScore := 0
			for _, s := range history {
				if s > maxScore {
					maxScore = s
				}
			}
			if maxScore == 0 {
				maxScore = 1 // Prevent division by zero
			}

			// Draw graph lines
			numPoints := len(history)
			if numPoints > 1 {
				pointSpacing := float32(graphWidth) / float32(numPoints-1)
				for i := 0; i < numPoints-1; i++ {
					x1 := float32(graphX) + pointSpacing*float32(i)
					y1 := float32(graphY) + float32(graphHeight) - (float32(history[i])/float32(maxScore))*float32(graphHeight)
					x2 := float32(graphX) + pointSpacing*float32(i+1)
					y2 := float32(graphY) + float32(graphHeight) - (float32(history[i+1])/float32(maxScore))*float32(graphHeight)
					rl.DrawLineEx(
						rl.Vector2{X: x1, Y: y1},
						rl.Vector2{X: x2, Y: y2},
						2,
						rl.Green)
				}
			}

			// Draw axis labels
			rl.DrawText("Games", graphX+graphWidth/2-30, graphY+graphHeight+5, fontSize, rl.White)
			rl.DrawText("Score", graphX-35, graphY+graphHeight/2-10, fontSize, rl.White)
		}
	}

	// Draw shared food
	for _, food := range g.GetStateManager().GetFoodList() {
		rl.DrawRectangle(
			r.offsetX+int32(food.X*int(r.cellSize)),
			r.offsetY+int32(food.Y*int(r.cellSize)),
			r.cellSize, r.cellSize, rl.Red)
	}

	// Draw game over text if snake is dead
	if len(snakes) > 0 && snakes[0].Dead {
		gameOverText := "Game Over! (Restarting...)"
		textWidth := rl.MeasureText(gameOverText, fontSize)
		rl.DrawText(gameOverText,
			r.offsetX+(r.totalGridWidth-int32(textWidth))/2,
			r.offsetY+r.totalGridHeight/2,
			fontSize, rl.White)
	}
	rl.EndDrawing()
}
