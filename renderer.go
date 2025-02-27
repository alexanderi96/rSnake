package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const borderPadding = 0 // No padding

// Colori personalizzati per le statistiche
var (
	scoreColor       = rl.Color{R: 0, G: 180, B: 0, A: 255}     // Verde scuro
	avgScoreColor    = rl.Color{R: 144, G: 238, B: 144, A: 255} // Verde chiaro
	durationColor    = rl.Color{R: 128, G: 0, B: 128, A: 255}   // Viola scuro
	avgDurationColor = rl.Color{R: 216, G: 191, B: 216, A: 255} // Viola chiaro
	epsilonColor     = rl.Color{R: 255, G: 165, B: 0, A: 255}   // Arancione
)

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

func (r *Renderer) Draw(g *Game) {
	r.UpdateDimensions()
	rl.BeginDrawing()
	rl.ClearBackground(rl.Gray)

	fontSize := int32(r.screenHeight / 45) // Dynamic font size

	// Use full screen for grid
	availableWidth := r.screenWidth
	availableHeight := r.screenHeight

	// Calculate cell size based on available space and grid dimensions
	cellW := availableWidth / int32(g.Grid.Width)
	cellH := availableHeight / int32(g.Grid.Height)
	r.cellSize = min(cellW, cellH)

	// Calculate total grid dimensions
	r.totalGridWidth = r.cellSize * int32(g.Grid.Width)
	r.totalGridHeight = r.cellSize * int32(g.Grid.Height)

	// Position grid at the top left corner
	r.offsetX = 0
	r.offsetY = 0

	// Draw grid
	gridColor := rl.Color{R: 100, G: 100, B: 100, A: 100} // Colore grigio chiaro semi-trasparente
	for x := int32(0); x <= int32(g.Grid.Width); x++ {
		rl.DrawLineV(
			rl.Vector2{X: float32(r.offsetX + x*r.cellSize), Y: float32(r.offsetY)},
			rl.Vector2{X: float32(r.offsetX + x*r.cellSize), Y: float32(r.offsetY + r.totalGridHeight)},
			gridColor,
		)
	}
	for y := int32(0); y <= int32(g.Grid.Height); y++ {
		rl.DrawLineV(
			rl.Vector2{X: float32(r.offsetX), Y: float32(r.offsetY + y*r.cellSize)},
			rl.Vector2{X: float32(r.offsetX + r.totalGridWidth), Y: float32(r.offsetY + y*r.cellSize)},
			gridColor,
		)
	}

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
	if len(stats) == 0 {
		return
	}

	// Define graph dimensions
	graphHeight := int32(150)
	graphY := r.screenHeight - graphHeight

	// Draw stats vertically above the graph
	yOffset := graphY - int32(fontSize*6) // Space for 5 lines of stats plus padding
	xOffset := r.offsetX + int32(10)      // Small left padding
	lineSpacing := fontSize + 5           // Space between lines

	// Score
	scoreLabel := fmt.Sprintf("Score: %d", score)
	rl.DrawText(scoreLabel, xOffset, yOffset, fontSize, rl.White)
	yOffset += lineSpacing

	// Total games
	rl.DrawText(fmt.Sprintf("Total Games: %d", r.stats.TotalGames),
		xOffset,
		yOffset,
		fontSize, rl.White)
	yOffset += lineSpacing

	// Max score and its average (verde)
	maxScore := r.stats.GetAbsoluteMaxScore()
	rl.DrawText(fmt.Sprintf("Max Score: %d", maxScore),
		xOffset,
		yOffset,
		fontSize, scoreColor)
	yOffset += lineSpacing

	// Get the latest game record for averages
	latestGame := stats[len(stats)-1]

	// Average max score (verde chiaro)
	rl.DrawText(fmt.Sprintf("Avg Max: %.1f", latestGame.AverageMaxScore),
		xOffset,
		yOffset,
		fontSize, avgScoreColor)
	yOffset += lineSpacing

	// Average max duration (viola chiaro)
	rl.DrawText(fmt.Sprintf("Avg Max Time: %.1fs", latestGame.AverageMaxDuration),
		xOffset,
		yOffset,
		fontSize, avgDurationColor)
	yOffset += lineSpacing

	// Current epsilon value (arancione)
	rl.DrawText(fmt.Sprintf("Epsilon: %.3f", latestGame.Epsilon),
		xOffset,
		yOffset,
		fontSize, epsilonColor)

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
	// Graph dimensions are now defined in Draw()
	graphHeight := int32(150)
	graphWidth := r.screenWidth
	graphY := r.screenHeight - graphHeight

	// Get stats data
	stats := r.stats.GetStats()
	if len(stats) == 0 {
		return
	}

	// Find max values for scaling
	maxScore := r.stats.GetAbsoluteMaxScore()
	var maxDuration float64
	for _, game := range stats {
		if game.MaxDuration > maxDuration {
			maxDuration = game.MaxDuration
		}
	}

	// Sort stats by timestamp
	sortedStats := make([]GameRecord, len(stats))
	copy(sortedStats, stats)
	for i := 0; i < len(sortedStats)-1; i++ {
		for j := i + 1; j < len(sortedStats); j++ {
			if sortedStats[i].StartTime.After(sortedStats[j].StartTime) {
				sortedStats[i], sortedStats[j] = sortedStats[j], sortedStats[i]
			}
		}
	}

	scaleY := float32(graphHeight-40) / float32(maxScore)
	durationScaleY := float32(graphHeight-40) / float32(maxDuration)

	if len(sortedStats) == 0 {
		return
	}

	// Calculate point spacing using all available width
	pointSpacing := float32(graphWidth) / float32(len(sortedStats)-1)
	currentX := float32(borderPadding)

	const (
		barWidth   = float32(4) // Standard width for all bars
		barSpacing = float32(6) // Space between score and duration bars
		barAlpha   = uint8(180) // Standard alpha for bars
	)

	// Draw all records chronologically
	for i, game := range sortedStats {
		// Calculate x positions for score and duration bars
		scoreX := currentX - barSpacing/2
		durationX := currentX + barSpacing/2

		// Draw score bar and point (green)
		var scoreY float32
		if game.CompressionIndex == 0 {
			scoreY = float32(graphY+graphHeight) - float32(game.Score)*scaleY
			// Draw single score bar
			rl.DrawRectangle(
				int32(scoreX-barWidth/2),
				int32(scoreY),
				int32(barWidth),
				int32(float32(graphY+graphHeight)-scoreY),
				scoreColor)
		} else {
			scoreY = float32(graphY+graphHeight) - float32(game.CompressedAverageScore)*scaleY
			minScoreY := float32(graphY+graphHeight) - float32(game.MinScore)*scaleY
			maxScoreY := float32(graphY+graphHeight) - float32(game.MaxScore)*scaleY

			// Draw min-max range bar for compressed records
			rl.DrawRectangle(
				int32(scoreX-barWidth/2),
				int32(maxScoreY),
				int32(barWidth),
				int32(minScoreY-maxScoreY),
				scoreColor)

			// Draw compressed average score marker
			rl.DrawRectangle(
				int32(scoreX-barWidth/2),
				int32(scoreY),
				int32(barWidth),
				int32(2),
				scoreColor)
		}

		// Draw average max score point and line (verde chiaro)
		avgMaxScoreY := float32(graphY+graphHeight) - float32(game.AverageMaxScore)*scaleY
		rl.DrawCircle(
			int32(scoreX),
			int32(avgMaxScoreY),
			2,
			avgScoreColor)

		// Draw duration bar and point (purple)
		var durationY float32
		if game.CompressionIndex == 0 {
			duration := float32(game.EndTime.Sub(game.StartTime).Seconds())
			durationY = float32(graphY+graphHeight) - duration*durationScaleY
			// Draw single duration bar
			rl.DrawRectangle(
				int32(durationX-barWidth/2),
				int32(durationY),
				int32(barWidth),
				int32(float32(graphY+graphHeight)-durationY),
				durationColor)
		} else {
			durationY = float32(graphY+graphHeight) - float32(game.CompressedAverageDuration)*durationScaleY
			minDurationY := float32(graphY+graphHeight) - float32(game.MinDuration)*durationScaleY
			maxDurationY := float32(graphY+graphHeight) - float32(game.MaxDuration)*durationScaleY

			// Draw min-max range bar for compressed records
			rl.DrawRectangle(
				int32(durationX-barWidth/2),
				int32(maxDurationY),
				int32(barWidth),
				int32(minDurationY-maxDurationY),
				durationColor)

			// Draw compressed average duration marker
			rl.DrawRectangle(
				int32(durationX-barWidth/2),
				int32(durationY),
				int32(barWidth),
				int32(2),
				durationColor)
		}

		// Draw average max duration point and line (viola chiaro)
		avgMaxDurationY := float32(graphY+graphHeight) - float32(game.AverageMaxDuration)*durationScaleY
		rl.DrawCircle(
			int32(durationX),
			int32(avgMaxDurationY),
			2,
			avgDurationColor)

		// Draw connecting lines to next point if not the last point
		if i < len(sortedStats)-1 {
			nextX := currentX + pointSpacing
			nextGame := sortedStats[i+1]

			// Draw connecting lines for compressed scores and durations
			var nextScoreY, nextDurationY float32
			if nextGame.CompressionIndex == 0 {
				nextScoreY = float32(graphY+graphHeight) - float32(nextGame.Score)*scaleY
				nextDuration := float32(nextGame.EndTime.Sub(nextGame.StartTime).Seconds())
				nextDurationY = float32(graphY+graphHeight) - nextDuration*durationScaleY
			} else {
				nextScoreY = float32(graphY+graphHeight) - float32(nextGame.CompressedAverageScore)*scaleY
				nextDurationY = float32(graphY+graphHeight) - float32(nextGame.CompressedAverageDuration)*durationScaleY
			}

			rl.DrawLine(
				int32(scoreX),
				int32(scoreY),
				int32(nextX-barSpacing/2),
				int32(nextScoreY),
				scoreColor)
			rl.DrawLine(
				int32(durationX),
				int32(durationY),
				int32(nextX+barSpacing/2),
				int32(nextDurationY),
				durationColor)

			// Draw connecting lines for running averages
			nextAvgMaxScoreY := float32(graphY+graphHeight) - float32(nextGame.AverageMaxScore)*scaleY
			nextAvgMaxDurationY := float32(graphY+graphHeight) - float32(nextGame.AverageMaxDuration)*durationScaleY

			rl.DrawLine(
				int32(scoreX),
				int32(avgMaxScoreY),
				int32(nextX-barSpacing/2),
				int32(nextAvgMaxScoreY),
				avgScoreColor)
			rl.DrawLine(
				int32(durationX),
				int32(avgMaxDurationY),
				int32(nextX+barSpacing/2),
				int32(nextAvgMaxDurationY),
				avgDurationColor)

			// Draw epsilon line
			epsilonY := float32(graphY+graphHeight) - float32(game.Epsilon)*float32(graphHeight-40)
			nextEpsilonY := float32(graphY+graphHeight) - float32(nextGame.Epsilon)*float32(graphHeight-40)
			rl.DrawLine(
				int32(currentX),
				int32(epsilonY),
				int32(nextX),
				int32(nextEpsilonY),
				epsilonColor)
			// Draw epsilon point
			rl.DrawCircle(
				int32(currentX),
				int32(epsilonY),
				3,
				epsilonColor)
		}

		currentX += pointSpacing
	}
}
