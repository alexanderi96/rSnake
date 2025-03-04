package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const borderPadding = 0 // No padding

// Colori personalizzati per le statistiche
var (
	scoreColor    = rl.Color{R: 0, G: 180, B: 0, A: 255}   // Verde per il punteggio
	durationColor = rl.Color{R: 128, G: 0, B: 128, A: 255} // Viola per la durata
	entropyColor  = rl.Color{R: 255, G: 165, B: 0, A: 255} // Arancione per l'entropia
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
	numAgents       int   // Numero di agenti da visualizzare
	agentsPerRow    int   // Numero di agenti per riga
}

func NewRenderer() *Renderer {
	r := &Renderer{
		numAgents:    4, // Default a 4 agenti
		agentsPerRow: 2, // 2x2 grid layout
	}
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

// DrawMultiAgent disegna tutti gli agenti in una griglia
func (r *Renderer) DrawMultiAgent(trainingManager *TrainingManager) {
	if trainingManager == nil {
		return
	}

	r.UpdateDimensions()
	rl.BeginDrawing()
	rl.ClearBackground(rl.Gray)

	fontSize := int32(r.screenHeight / 45) // Dynamic font size

	// Calcola le dimensioni per ogni griglia di gioco
	gridMargin := int32(10)
	availableWidth := (r.screenWidth - gridMargin*int32(r.agentsPerRow+1)) / int32(r.agentsPerRow)
	numRows := (r.numAgents + r.agentsPerRow - 1) / r.agentsPerRow
	availableHeight := (r.screenHeight - gridMargin*int32(numRows+1)) / int32(numRows)

	// Ottieni il primo gioco per le dimensioni della griglia
	firstGame := trainingManager.GetGame(0)
	if firstGame == nil {
		rl.EndDrawing()
		return
	}

	// Calcola la dimensione delle celle basata sullo spazio disponibile
	cellW := availableWidth / int32(firstGame.Grid.Width)
	cellH := availableHeight / int32(firstGame.Grid.Height)
	r.cellSize = min(cellW, cellH)

	// Disegna ogni agente nella sua posizione nella griglia
	for i := 0; i < r.numAgents; i++ {
		game := trainingManager.GetGame(i)
		if game == nil {
			continue
		}

		// Calcola la posizione nella griglia
		row := i / r.agentsPerRow
		col := i % r.agentsPerRow

		// Calcola l'offset per questo agente
		r.offsetX = gridMargin + int32(col)*(availableWidth+gridMargin)
		r.offsetY = gridMargin + int32(row)*(availableHeight+gridMargin)

		// Calcola le dimensioni totali della griglia per questo agente
		r.totalGridWidth = r.cellSize * int32(game.Grid.Width)
		r.totalGridHeight = r.cellSize * int32(game.Grid.Height)

		// Disegna il numero dell'agente
		agentLabel := fmt.Sprintf("Agent %d", i+1)
		labelWidth := rl.MeasureText(agentLabel, fontSize)
		rl.DrawText(agentLabel,
			r.offsetX+(availableWidth-labelWidth)/2,
			r.offsetY-fontSize-5,
			fontSize,
			rl.White)

		r.drawGame(game)
	}

	// Disegna le statistiche globali nella parte inferiore
	if r.stats != nil {
		r.drawStatsGraph()
	}

	rl.EndDrawing()
}

// drawGame disegna un singolo gioco
func (r *Renderer) drawGame(g *Game) {
	if g == nil {
		return
	}

	r.game = g // Store game reference

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

	// Draw food
	food := g.GetFood()
	rl.DrawRectangle(
		r.offsetX+int32(food.X*int(r.cellSize)),
		r.offsetY+int32(food.Y*int(r.cellSize)),
		r.cellSize, r.cellSize, rl.Red)

	// Only draw stats and state grid if stats are initialized
	if r.stats != nil {
		allStats := r.stats.GetStats()
		if len(allStats) > 0 {
			// Take only the last stat for display
			latestGame := allStats[len(allStats)-1]
			fontSize := int32(r.screenHeight / 45) // Dynamic font size

			// Define graph dimensions
			graphHeight := int32(150)
			graphY := r.screenHeight - graphHeight

			// Get all stats values first
			maxScore := r.stats.GetAbsoluteMaxScore()
			gamesPerSecond := r.stats.GetGamesPerSecond()

			// Calculate max width needed for stats
			statsWidth := int32(0)
			statsTexts := []string{
				fmt.Sprintf("Score: %d", score),
				fmt.Sprintf("Total Games: %d", r.stats.TotalGames),
				fmt.Sprintf("Max Score: %d", maxScore),
				fmt.Sprintf("Policy Entropy: %.3f", latestGame.PolicyEntropy),
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

			// Current policy entropy value (arancione)
			rl.DrawText(fmt.Sprintf("Policy Entropy: %.3f", latestGame.PolicyEntropy),
				xOffset,
				yOffset,
				fontSize, entropyColor)
			yOffset += lineSpacing

			// Games per second (bianco)
			rl.DrawText(fmt.Sprintf("Games/s: %d", gamesPerSecond),
				xOffset,
				yOffset,
				fontSize, rl.White)

			// Draw dark overlay for graph
			rl.DrawRectangle(0, graphY, r.screenWidth, graphHeight, rl.Color{R: 0, G: 0, B: 0, A: 100})

			// Draw statistics graph
			r.drawStatsGraph()
		}
	}

	// Draw state visualization grid
	r.drawStateGrid(g)
}

// drawStateGrid draws the state information grids
func (r *Renderer) drawStateGrid(g *Game) {
	if g == nil {
		return
	}

	// Get state matrix (3x8)
	state := g.GetStateInfo()

	// Calculate grid positions and sizes
	gridSize := int32(25)                                    // Size of each cell in the grid
	gridSpacing := int32(35)                                 // Space between grids
	startX := r.screenWidth - (gridSize*3*4 + gridSpacing*3) // Position from right edge, accounting for 4 grids with spacing
	startY := int32(30)                                      // Start Y position
	fontSize := int32(14)                                    // Font size for labels

	// Define grid positions
	grids := []struct {
		label  string
		x      int32
		color  func(value float64) rl.Color
		values []float64
	}{
		{
			label:  "Combined Values",
			x:      startX,
			color:  r.getStateColor,
			values: nil, // Will be filled with combined values
		},
		{
			label: "Wall Values",
			x:     startX + gridSize*3 + gridSpacing,
			color: func(value float64) rl.Color {
				if value > 0 {
					return rl.Gray
				}
				return rl.Black
			},
			values: state[0:8], // Wall values
		},
		{
			label: "Body Values",
			x:     startX + gridSize*6 + gridSpacing*2,
			color: func(value float64) rl.Color {
				if value > 0 {
					return rl.Blue
				}
				return rl.Black
			},
			values: state[8:16], // Body values
		},
		{
			label: "Food Values",
			x:     startX + gridSize*9 + gridSpacing*3,
			color: func(value float64) rl.Color {
				if value > 0 {
					return rl.Red
				}
				return rl.Black
			},
			values: state[16:24], // Food values
		},
	}

	// Calculate directions
	currentDir := g.GetCurrentDirection()
	leftDir := currentDir.TurnLeft()
	rightDir := currentDir.TurnRight()

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

	// Draw each grid
	for gridIndex, grid := range grids {
		// Draw background
		rl.DrawRectangle(
			grid.x-5,
			startY-5,
			gridSize*3+10,
			gridSize*3+10,
			rl.Color{R: 0, G: 0, B: 0, A: 100},
		)

		// Draw label
		labelWidth := rl.MeasureText(grid.label, fontSize)
		labelX := grid.x + (gridSize*3-labelWidth)/2
		labelY := startY - fontSize - 2
		rl.DrawText(grid.label, labelX, labelY, fontSize, rl.White)

		// Draw the 3x3 grid
		for row := -1; row <= 1; row++ {
			for col := -1; col <= 1; col++ {
				if row == 0 && col == 0 {
					// Center cell (snake position)
					cellX := grid.x + gridSize
					cellY := startY + gridSize
					rl.DrawRectangle(cellX, cellY, gridSize, gridSize, rl.DarkGray)
					rl.DrawRectangleLines(cellX, cellY, gridSize, gridSize, rl.White)
					continue
				}

				cellX := grid.x + int32(col+1)*gridSize
				cellY := startY + int32(row+1)*gridSize

				// Get the direction for this position
				if dir, ok := dirMap[Point{X: col, Y: row}]; ok {
					var value float64
					if gridIndex == 0 {
						// Combined values grid
						value = g.GetCombinedDirectionalInfo(dir)
					} else {
						// Find the index in the direction array
						dirIndex := -1
						for i, d := range directions {
							if d == dir {
								dirIndex = i
								break
							}
						}
						if dirIndex >= 0 {
							value = grid.values[dirIndex]
						}
					}

					// Get color based on value
					color := grid.color(value)

					// Draw cell
					rl.DrawRectangle(cellX, cellY, gridSize, gridSize, color)
					rl.DrawRectangleLines(cellX, cellY, gridSize, gridSize, rl.White)

					// Draw value text
					valueText := fmt.Sprintf("%.2f", value)
					textWidth := rl.MeasureText(valueText, fontSize)
					textX := cellX + (gridSize-textWidth)/2
					textY := cellY + (gridSize-fontSize)/2
					rl.DrawText(valueText, textX, textY, fontSize, rl.Black)
				}
			}
		}
	}
}

func (r *Renderer) drawStatsGraph() {
	// Graph dimensions are now defined in Draw()
	graphHeight := int32(150)
	graphWidth := r.screenWidth
	graphY := r.screenHeight - graphHeight

	// Get stats data - only last 100
	allStats := r.stats.GetStats()
	if len(allStats) == 0 {
		return
	}

	// Take only the last 100 stats
	startIdx := len(allStats)
	if startIdx > 100 {
		startIdx = len(allStats) - 100
	}
	stats := allStats[startIdx:]

	// Find max values for scaling
	maxScore := r.stats.GetAbsoluteMaxScore()
	var maxDuration float64
	for _, game := range stats {
		duration := game.EndTime.Sub(game.StartTime).Seconds()
		if duration > maxDuration {
			maxDuration = duration
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

		// Calculate score and duration Y positions
		scoreY := float32(graphY+graphHeight) - float32(game.Score)*scaleY
		duration := float32(game.EndTime.Sub(game.StartTime).Seconds())
		durationY := float32(graphY+graphHeight) - duration*durationScaleY

		// Draw points for score and duration
		rl.DrawCircle(int32(scoreX), int32(scoreY), 2, scoreColor)
		rl.DrawCircle(int32(durationX), int32(durationY), 2, durationColor)

		// Draw connecting lines to next point if not the last point
		if i < len(sortedStats)-1 {
			nextX := currentX + pointSpacing
			nextGame := sortedStats[i+1]

			// Calculate next point positions
			nextScoreY := float32(graphY+graphHeight) - float32(nextGame.Score)*scaleY
			nextDuration := float32(nextGame.EndTime.Sub(nextGame.StartTime).Seconds())
			nextDurationY := float32(graphY+graphHeight) - nextDuration*durationScaleY

			// Draw connecting lines
			rl.DrawLine(int32(scoreX), int32(scoreY), int32(nextX-barSpacing/2), int32(nextScoreY), scoreColor)
			rl.DrawLine(int32(durationX), int32(durationY), int32(nextX+barSpacing/2), int32(nextDurationY), durationColor)

			// Draw policy entropy line
			entropyY := float32(graphY+graphHeight) - float32(game.PolicyEntropy)*float32(graphHeight-40)
			nextEntropyY := float32(graphY+graphHeight) - float32(nextGame.PolicyEntropy)*float32(graphHeight-40)
			rl.DrawLine(
				int32(currentX),
				int32(entropyY),
				int32(nextX),
				int32(nextEntropyY),
				entropyColor)
			// Draw policy entropy point
			rl.DrawCircle(
				int32(currentX),
				int32(entropyY),
				3,
				entropyColor)
		}

		currentX += pointSpacing
	}
}
