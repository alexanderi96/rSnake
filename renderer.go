package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const borderPadding = 0 // No padding

// Colori personalizzati per le statistiche
var (
	scoreColor          = rl.Color{R: 0, G: 180, B: 0, A: 255}     // Verde scuro
	avgScoreColor       = rl.Color{R: 144, G: 238, B: 144, A: 255} // Verde chiaro
	durationColor       = rl.Color{R: 128, G: 0, B: 128, A: 255}   // Viola scuro
	avgDurationColor    = rl.Color{R: 216, G: 191, B: 216, A: 255} // Viola chiaro
	epsilonColor        = rl.Color{R: 255, G: 165, B: 0, A: 255}   // Arancione
	compressionIndColor = rl.Color{R: 255, G: 0, B: 0, A: 255}     // Rosso per indicatori di compressione
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
	game            *Game // Add game reference to access state info
}

func NewRenderer() *Renderer {
	r := &Renderer{}
	r.UpdateDimensions()
	return r
}

// getStateColor returns a color based on a state value between -1 and 1
func (r *Renderer) getStateColor(value float64) rl.Color {
	// Ensure value is between -1 and 1
	if value < -1 {
		value = -1
	} else if value > 1 {
		value = 1
	}

	// For negative values: interpolate between red and yellow
	// For positive values: interpolate between yellow and green
	if value < 0 {
		// From red (255,0,0) to yellow (255,255,0)
		green := uint8((value + 1) * 255) // value + 1 maps [-1,0] to [0,1]
		return rl.Color{R: 255, G: green, B: 0, A: 255}
	} else {
		// From yellow (255,255,0) to green (0,255,0)
		red := uint8((1 - value) * 255) // 1 - value maps [0,1] to [1,0]
		return rl.Color{R: red, G: 255, B: 0, A: 255}
	}
}

// drawStateGrids draws both the state matrix and directional values grid
func (r *Renderer) drawStateGrid(g *Game) {
	if g == nil {
		return
	}

	// Get state matrix (3x8)
	state := g.GetStateInfo()

	// Calculate grid positions
	gridSize := int32(25)                    // Size of each cell in the grid
	gridX := r.screenWidth - gridSize*8 - 10 // Position from right edge accounting for 8 cells width
	gridY := int32(5)                        // Same Y as stats start

	// Calculate position for directional grid (3x3)
	dirGridSize := int32(30)                                    // Slightly larger for better visibility
	dirGridX := r.screenWidth - gridSize*8 - dirGridSize*3 - 30 // Position to the left of state grid
	dirGridY := gridY                                           // Same Y as state grid

	// Draw backgrounds
	// For state grid
	rl.DrawRectangle(
		gridX-5,
		gridY-5,
		gridSize*8+10, // Width for 8 cells
		gridSize*3+10, // Height for 3 cells
		rl.Color{R: 0, G: 0, B: 0, A: 100},
	)

	// For directional grid
	rl.DrawRectangle(
		dirGridX-5,
		dirGridY-5,
		dirGridSize*3+10, // Width for 3 cells
		dirGridSize*3+10, // Height for 3 cells
		rl.Color{R: 0, G: 0, B: 0, A: 100},
	)

	// Draw labels
	fontSize := int32(12)

	// For state grid
	labels := []string{"BL", "L", "FL", "F", "FR", "R", "BR", "B"}
	rowLabels := []string{"Wall", "Body", "Food"}

	// Draw state grid column labels
	for i, label := range labels {
		labelWidth := rl.MeasureText(label, fontSize)
		labelX := gridX + int32(i)*gridSize + (gridSize-labelWidth)/2
		labelY := gridY - fontSize - 2
		rl.DrawText(label, labelX, labelY, fontSize, rl.White)
	}

	// Draw state grid row labels
	for i, label := range rowLabels {
		labelWidth := rl.MeasureText(label, fontSize)
		labelX := gridX - labelWidth - 5
		labelY := gridY + int32(i)*gridSize + (gridSize-fontSize)/2
		rl.DrawText(label, labelX, labelY, fontSize, rl.White)
	}

	// Draw "Combined Values" label for directional grid
	dirLabel := "Combined Values"
	labelWidth := rl.MeasureText(dirLabel, fontSize)
	labelX := dirGridX + (dirGridSize*3-labelWidth)/2
	labelY := dirGridY - fontSize - 2
	rl.DrawText(dirLabel, labelX, labelY, fontSize, rl.White)

	// Draw the state grid cells
	for row := 0; row < 3; row++ {
		for col := 0; col < 8; col++ {
			value := state[row*8+col]
			cellX := gridX + int32(col)*gridSize
			cellY := gridY + int32(row)*gridSize

			// Choose color based on the type of information and value
			var color rl.Color
			switch row {
			case 0: // Walls
				if value > 0 {
					color = rl.Gray
				} else {
					color = rl.Black
				}
			case 1: // Body
				if value > 0 {
					color = rl.Blue
				} else {
					color = rl.Black
				}
			case 2: // Food
				if value > 0 {
					color = rl.Red
				} else {
					color = rl.Black
				}
			}

			// Draw cell
			rl.DrawRectangle(cellX, cellY, gridSize, gridSize, color)
			rl.DrawRectangleLines(cellX, cellY, gridSize, gridSize, rl.White)
		}
	}

	// Draw the directional grid (3x3)
	currentDir := g.GetCurrentDirection()
	leftDir := currentDir.TurnLeft()
	rightDir := currentDir.TurnRight()

	// Calculate all 8 directions around the snake
	directions := []Point{
		{X: -currentDir.ToPoint().X + leftDir.ToPoint().X, Y: -currentDir.ToPoint().Y + leftDir.ToPoint().Y}, // backLeft
		leftDir.ToPoint(), // left
		{X: currentDir.ToPoint().X + leftDir.ToPoint().X, Y: currentDir.ToPoint().Y + leftDir.ToPoint().Y}, // frontLeft
		currentDir.ToPoint(), // front
		{X: currentDir.ToPoint().X + rightDir.ToPoint().X, Y: currentDir.ToPoint().Y + rightDir.ToPoint().Y}, // frontRight
		rightDir.ToPoint(), // right
		{X: -currentDir.ToPoint().X + rightDir.ToPoint().X, Y: -currentDir.ToPoint().Y + rightDir.ToPoint().Y}, // backRight
		{X: -currentDir.ToPoint().X, Y: -currentDir.ToPoint().Y},                                               // back
	}

	// Map of positions to directions
	dirMap := map[Point]Point{
		{X: -1, Y: -1}: directions[0], // backLeft
		{X: -1, Y: 0}:  directions[1], // left
		{X: -1, Y: 1}:  directions[2], // frontLeft
		{X: 0, Y: -1}:  directions[7], // back
		{X: 0, Y: 1}:   directions[3], // front
		{X: 1, Y: -1}:  directions[6], // backRight
		{X: 1, Y: 0}:   directions[5], // right
		{X: 1, Y: 1}:   directions[4], // frontRight
	}

	// Draw the 3x3 grid
	for row := -1; row <= 1; row++ {
		for col := -1; col <= 1; col++ {
			if row == 0 && col == 0 {
				// Center cell (snake position)
				cellX := dirGridX + dirGridSize
				cellY := dirGridY + dirGridSize
				rl.DrawRectangle(cellX, cellY, dirGridSize, dirGridSize, rl.DarkGray)
				rl.DrawRectangleLines(cellX, cellY, dirGridSize, dirGridSize, rl.White)
				continue
			}

			cellX := dirGridX + int32(col+1)*dirGridSize
			cellY := dirGridY + int32(row+1)*dirGridSize

			// Get the direction for this position
			if dir, ok := dirMap[Point{X: col, Y: row}]; ok {
				// Get combined directional value
				value := g.GetCombinedDirectionalInfo(dir)
				// Get color based on value
				color := r.getStateColor(value)

				// Draw cell with value
				rl.DrawRectangle(cellX, cellY, dirGridSize, dirGridSize, color)
				rl.DrawRectangleLines(cellX, cellY, dirGridSize, dirGridSize, rl.White)

				// Draw value text
				valueText := fmt.Sprintf("%.2f", value)
				textWidth := rl.MeasureText(valueText, fontSize)
				textX := cellX + (dirGridSize-textWidth)/2
				textY := cellY + (dirGridSize-fontSize)/2
				rl.DrawText(valueText, textX, textY, fontSize, rl.Black)
			}
		}
	}
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

// drawPlayableArea disegna un rettangolo che evidenzia l'area giocabile corrente
func (r *Renderer) drawPlayableArea(g *Game) {
	// Ottieni i limiti dell'area giocabile
	minX, maxX, minY, maxY := g.getPlayableArea()

	// Calcola le coordinate sullo schermo
	screenMinX := r.offsetX + int32(minX*int(r.cellSize))
	screenMinY := r.offsetY + int32(minY*int(r.cellSize))
	screenWidth := int32((maxX - minX + 1) * int(r.cellSize))
	screenHeight := int32((maxY - minY + 1) * int(r.cellSize))

	// Disegna un rettangolo semi-trasparente per evidenziare l'area
	areaColor := rl.Color{R: 255, G: 255, B: 255, A: 30} // Bianco semi-trasparente
	rl.DrawRectangle(screenMinX, screenMinY, screenWidth, screenHeight, areaColor)

	// Disegna il bordo dell'area
	borderColor := rl.Color{R: 255, G: 255, B: 255, A: 100} // Bianco più visibile per il bordo
	rl.DrawRectangleLines(screenMinX, screenMinY, screenWidth, screenHeight, borderColor)
}

func (r *Renderer) Draw(g *Game) {
	r.game = g // Store game reference
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

	// Draw playable area
	r.drawPlayableArea(g)
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

	// Get all stats values first
	maxScore := r.stats.GetAbsoluteMaxScore()
	latestGame := stats[len(stats)-1]
	gamesPerSecond := r.stats.GetGamesPerSecond()

	// Calculate max width needed for stats
	statsWidth := int32(0)
	statsTexts := []string{
		fmt.Sprintf("Score: %d", score),
		fmt.Sprintf("Total Games: %d", r.stats.TotalGames),
		fmt.Sprintf("Max Score: %d", maxScore),
		fmt.Sprintf("Epsilon: %.3f", latestGame.Epsilon),
		fmt.Sprintf("Games/s: %d", gamesPerSecond),
	}

	for _, text := range statsTexts {
		width := rl.MeasureText(text, fontSize)
		if width > statsWidth {
			statsWidth = width
		}
	}

	// Add padding to width and height
	statsWidth += 20                      // 10px padding on each side
	statsHeight := int32(fontSize*5 + 20) // Height for stats area plus padding

	// Draw dark overlay for stats at the top
	rl.DrawRectangle(0, 0, statsWidth, statsHeight, rl.Color{R: 0, G: 0, B: 0, A: 100})

	// Draw stats at the top of the screen
	yOffset := int32(10)             // Start from top with padding
	xOffset := r.offsetX + int32(10) // Small left padding
	lineSpacing := fontSize + 5      // Space between lines

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
	rl.DrawText(fmt.Sprintf("Max Score: %d", maxScore),
		xOffset,
		yOffset,
		fontSize, scoreColor)
	yOffset += lineSpacing

	// Current epsilon value (arancione)
	rl.DrawText(fmt.Sprintf("Epsilon: %.3f", latestGame.Epsilon),
		xOffset,
		yOffset,
		fontSize, epsilonColor)
	yOffset += lineSpacing

	// Games per second (bianco)
	rl.DrawText(fmt.Sprintf("Games/s: %d", gamesPerSecond),
		xOffset,
		yOffset,
		fontSize, rl.White)

	// Draw food
	food := g.GetFood()
	rl.DrawRectangle(
		r.offsetX+int32(food.X*int(r.cellSize)),
		r.offsetY+int32(food.Y*int(r.cellSize)),
		r.cellSize, r.cellSize, rl.Red)

	// Draw dark overlay for graph
	rl.DrawRectangle(0, graphY, r.screenWidth, graphHeight, rl.Color{R: 0, G: 0, B: 0, A: 100})

	// Draw statistics graph
	r.drawStatsGraph()

	// Draw state grid after stats but before ending drawing
	r.drawStateGrid(g)
	rl.EndDrawing()
}

// drawCompressionIndicator disegna una barra verticale rossa con l'indicatore del livello di compressione
func (r *Renderer) drawCompressionIndicator(x float32, graphY, graphHeight int32, compressionIndex int) {
	const (
		indicatorWidth = 2 // Larghezza della barra verticale
		textPadding    = 5 // Padding sopra il grafico per il testo
	)

	// Disegna la barra verticale
	rl.DrawRectangle(
		int32(x-indicatorWidth/2),
		graphY,
		indicatorWidth,
		graphHeight,
		compressionIndColor,
	)

	// Mostra direttamente il livello di compressione
	compressionText := fmt.Sprintf("L%d", compressionIndex)
	fontSize := int32(20)
	textWidth := rl.MeasureText(compressionText, fontSize)

	// Posiziona il testo centrato sopra la barra
	rl.DrawText(
		compressionText,
		int32(x)-textWidth/2,
		graphY-fontSize-textPadding,
		fontSize,
		compressionIndColor,
	)
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

	// Draw compression indicators first (so they appear behind the data)
	for i, game := range sortedStats {
		if game.CompressionIndex > 0 {
			// Verifica se è l'ultimo elemento del suo livello di compressione
			isLast := true
			if i < len(sortedStats)-1 {
				isLast = sortedStats[i+1].CompressionIndex != game.CompressionIndex
			}

			// Se è l'ultimo elemento, disegna l'indicatore alla posizione dell'elemento successivo
			if isLast && i < len(sortedStats)-1 {
				nextX := currentX + pointSpacing
				r.drawCompressionIndicator(nextX, graphY, graphHeight, game.CompressionIndex)
			}
		}
		currentX += pointSpacing
	}

	// Reset currentX for drawing the data
	currentX = float32(borderPadding)

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
