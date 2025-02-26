package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const borderPadding = 10 // Padding around game area

// Colori personalizzati per le statistiche
var (
	scoreColor       = rl.Color{R: 0, G: 180, B: 0, A: 255}     // Verde scuro
	avgScoreColor    = rl.Color{R: 144, G: 238, B: 144, A: 255} // Verde chiaro
	durationColor    = rl.Color{R: 128, G: 0, B: 128, A: 255}   // Viola scuro
	avgDurationColor = rl.Color{R: 216, G: 191, B: 216, A: 255} // Viola chiaro
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
	if len(stats) == 0 {
		return
	}

	// Draw stats with fixed spacing
	yOffset := r.offsetY + r.totalGridHeight + borderPadding
	xOffset := r.offsetX + 10
	spacing := int32(180) // Fixed spacing between stats

	// Score
	scoreLabel := fmt.Sprintf("Score: %d", score)
	rl.DrawText(scoreLabel, xOffset, yOffset, fontSize, rl.White)
	xOffset += spacing

	// Total games
	rl.DrawText(fmt.Sprintf("Total Games: %d", r.stats.TotalGames),
		xOffset,
		yOffset,
		fontSize, rl.White)
	xOffset += spacing

	// Running average score (verde chiaro)
	runningAvgScore := float64(r.stats.TotalScore) / float64(r.stats.TotalGames)
	rl.DrawText(fmt.Sprintf("Avg Score: %.1f", runningAvgScore),
		xOffset,
		yOffset,
		fontSize, avgScoreColor)
	xOffset += spacing

	// Max score (using absolute maximum across all compression levels)
	maxScore := r.stats.GetAbsoluteMaxScore()
	rl.DrawText(fmt.Sprintf("Max Score: %d", maxScore),
		xOffset,
		yOffset,
		fontSize, scoreColor)
	xOffset += spacing

	// Running average duration (viola chiaro)
	runningAvgDuration := r.stats.TotalTime / float64(r.stats.TotalGames)
	rl.DrawText(fmt.Sprintf("Avg Duration: %.1fs", runningAvgDuration),
		xOffset,
		yOffset,
		fontSize, avgDurationColor)

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

		// Draw running average score point and line (verde chiaro)
		runningScoreY := float32(graphY+graphHeight) - float32(game.RunningAverageScore)*scaleY
		rl.DrawCircle(
			int32(scoreX),
			int32(runningScoreY),
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

		// Draw running average duration point and line (viola chiaro)
		runningDurationY := float32(graphY+graphHeight) - float32(game.RunningAverageDuration)*durationScaleY
		rl.DrawCircle(
			int32(durationX),
			int32(runningDurationY),
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
			nextRunningScoreY := float32(graphY+graphHeight) - float32(nextGame.RunningAverageScore)*scaleY
			nextRunningDurationY := float32(graphY+graphHeight) - float32(nextGame.RunningAverageDuration)*durationScaleY

			rl.DrawLine(
				int32(scoreX),
				int32(runningScoreY),
				int32(nextX-barSpacing/2),
				int32(nextRunningScoreY),
				avgScoreColor)
			rl.DrawLine(
				int32(durationX),
				int32(runningDurationY),
				int32(nextX+barSpacing/2),
				int32(nextRunningDurationY),
				avgDurationColor)
		}

		currentX += pointSpacing
	}
}
