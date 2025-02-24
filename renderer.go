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
	totalGames := 0
	for _, game := range stats {
		totalGames += game.GamesCount
	}
	rl.DrawText(fmt.Sprintf("Total Games: %d", totalGames),
		xOffset,
		yOffset,
		fontSize, rl.White)
	xOffset += spacing

	// Average score
	avgScore := r.stats.GetAverageScore()
	rl.DrawText(fmt.Sprintf("Avg Score: %.1f", avgScore),
		xOffset,
		yOffset,
		fontSize, rl.Green)
	xOffset += spacing

	// Max score
	maxScore := r.stats.GetMaxScore()
	rl.DrawText(fmt.Sprintf("Max Score: %d", maxScore),
		xOffset,
		yOffset,
		fontSize, rl.Green)
	xOffset += spacing

	// Average duration
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
	if len(stats) == 0 {
		return
	}

	// Find max values for scaling
	var maxScore int
	var maxDuration float64

	for _, game := range stats {
		if game.MaxScore > maxScore {
			maxScore = game.MaxScore
		}
		if game.MaxDuration > maxDuration {
			maxDuration = game.MaxDuration
		}
	}

	// Sort stats by compression index (higher compression on the left)
	sortedStats := make([]GameRecord, len(stats))
	copy(sortedStats, stats)
	for i := 0; i < len(sortedStats)-1; i++ {
		for j := i + 1; j < len(sortedStats); j++ {
			if sortedStats[i].CompressionIndex < sortedStats[j].CompressionIndex {
				sortedStats[i], sortedStats[j] = sortedStats[j], sortedStats[i]
			}
		}
	}

	// Find max compression index to determine number of sectors
	maxCompressionIndex := 0
	for _, stat := range sortedStats {
		if stat.CompressionIndex > maxCompressionIndex {
			maxCompressionIndex = stat.CompressionIndex
		}
	}

	// Group stats by compression index
	groupedStats := make(map[int][]GameRecord)
	for _, stat := range sortedStats {
		groupedStats[stat.CompressionIndex] = append(groupedStats[stat.CompressionIndex], stat)
	}

	scaleY := float32(graphHeight-40) / float32(maxScore)
	durationScaleY := float32(graphHeight-40) / float32(maxDuration)

	// Count total records to determine spacing
	totalRecords := 0
	for compressionIdx := maxCompressionIndex; compressionIdx >= 0; compressionIdx-- {
		if records, exists := groupedStats[compressionIdx]; exists {
			totalRecords += len(records)
		}
	}

	if totalRecords == 0 {
		return
	}

	// Calculate point spacing for continuous layout
	pointSpacing := float32(graphWidth) / float32(totalRecords-1)
	currentX := float32(borderPadding)

	// Process compression indices in descending order
	for compressionIdx := maxCompressionIndex; compressionIdx >= 0; compressionIdx-- {
		records := groupedStats[compressionIdx]
		if len(records) == 0 {
			continue
		}
		// Draw all records in this group
		for i, game := range records {
			// Calculate bar width based on compression index
			barWidth := float32(4) // Base width for non-compressed records
			if game.CompressionIndex > 0 {
				barWidth = float32(4 + (game.CompressionIndex * 2)) // Increase width with compression level
			}

			// Draw score bar (green)
			var scoreY float32
			if game.CompressionIndex == 0 {
				scoreY = float32(graphY+graphHeight) - float32(game.Score)*scaleY
			} else {
				scoreY = float32(graphY+graphHeight) - float32(game.AverageScore)*scaleY
				minScoreY := float32(graphY+graphHeight) - float32(game.MinScore)*scaleY
				maxScoreY := float32(graphY+graphHeight) - float32(game.MaxScore)*scaleY

				// Draw min-max range bar for compressed records
				rl.DrawRectangle(
					int32(currentX-barWidth/2),
					int32(maxScoreY),
					int32(barWidth),
					int32(minScoreY-maxScoreY),
					rl.Color{R: 0, G: 180, B: 0, A: 100})
			}

			// Draw score point
			rl.DrawCircle(int32(currentX), int32(scoreY), float32(barWidth)/2, rl.Green)

			// Draw duration bar (purple)
			var durationY float32
			if game.CompressionIndex == 0 {
				duration := float32(game.EndTime.Sub(game.StartTime).Seconds())
				durationY = float32(graphY+graphHeight) - duration*durationScaleY
			} else {
				durationY = float32(graphY+graphHeight) - float32(game.AverageDuration)*durationScaleY
				minDurationY := float32(graphY+graphHeight) - float32(game.MinDuration)*durationScaleY
				maxDurationY := float32(graphY+graphHeight) - float32(game.MaxDuration)*durationScaleY

				// Draw min-max range bar for compressed records
				rl.DrawRectangle(
					int32(currentX-barWidth/2),
					int32(maxDurationY),
					int32(barWidth),
					int32(minDurationY-maxDurationY),
					rl.Color{R: 180, G: 0, B: 180, A: 100})
			}

			// Draw duration point
			rl.DrawCircle(int32(currentX), int32(durationY), float32(barWidth)/2, rl.Purple)

			// Store point data for line drawing
			if i == len(records)-1 && compressionIdx > 0 {
				// If this is the last point in a group and there are lower compression groups,
				// look for the first point in the next non-empty group
				nextCompressionIdx := compressionIdx - 1
				for nextCompressionIdx >= 0 {
					if nextRecords, exists := groupedStats[nextCompressionIdx]; exists && len(nextRecords) > 0 {
						nextX := currentX + pointSpacing
						var nextScoreY, nextDurationY float32
						nextGame := nextRecords[0]

						if nextGame.CompressionIndex == 0 {
							nextScoreY = float32(graphY+graphHeight) - float32(nextGame.Score)*scaleY
							nextDuration := float32(nextGame.EndTime.Sub(nextGame.StartTime).Seconds())
							nextDurationY = float32(graphY+graphHeight) - nextDuration*durationScaleY
						} else {
							nextScoreY = float32(graphY+graphHeight) - float32(nextGame.AverageScore)*scaleY
							nextDurationY = float32(graphY+graphHeight) - float32(nextGame.AverageDuration)*durationScaleY
						}

						rl.DrawLine(int32(currentX), int32(scoreY), int32(nextX), int32(nextScoreY), rl.Green)
						rl.DrawLine(int32(currentX), int32(durationY), int32(nextX), int32(nextDurationY), rl.Purple)
						break
					}
					nextCompressionIdx--
				}
			} else if i < len(records)-1 {
				// Draw line to next point in same group
				nextX := currentX + pointSpacing
				var nextScoreY, nextDurationY float32
				nextGame := records[i+1]

				if nextGame.CompressionIndex == 0 {
					nextScoreY = float32(graphY+graphHeight) - float32(nextGame.Score)*scaleY
					nextDuration := float32(nextGame.EndTime.Sub(nextGame.StartTime).Seconds())
					nextDurationY = float32(graphY+graphHeight) - nextDuration*durationScaleY
				} else {
					nextScoreY = float32(graphY+graphHeight) - float32(nextGame.AverageScore)*scaleY
					nextDurationY = float32(graphY+graphHeight) - float32(nextGame.AverageDuration)*durationScaleY
				}

				rl.DrawLine(int32(currentX), int32(scoreY), int32(nextX), int32(nextScoreY), rl.Green)
				rl.DrawLine(int32(currentX), int32(durationY), int32(nextX), int32(nextDurationY), rl.Purple)
			}

			currentX += pointSpacing
		}
	}
}
