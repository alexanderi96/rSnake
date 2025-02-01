package game

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"snake-game/ai"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	MaxAgents       = 4         // Maximum number of agents allowed
	FoodSpawnCycles = 100       // Number of game cycles between food spawns
	NumAgents       = MaxAgents // Fixed number of agents
	MaxPopulation   = 12        // Maximum allowed population before termination
	PopGrowthWindow = 100       // Window of cycles to measure population growth
)

var foodSpawnCounter = 0 // Counter for food spawn cycles

// Grid dimensions are now part of Game struct to be dynamic
type Grid struct {
	Width  int
	Height int
}

type GameStats struct {
	UUID           string       `json:"uuid"`
	PreviousGameID string       `json:"previous_game_id"`
	StartTime      time.Time    `json:"start_time"`
	EndTime        time.Time    `json:"end_time"`
	GameDuration   float64      `json:"game_duration"`
	Iterations     int          `json:"iterations"`
	AgentStats     []AgentStats `json:"agent_stats"`
}

type AgentStats struct {
	UUID         string  `json:"uuid"`
	Score        int     `json:"score"`
	AverageScore float64 `json:"average_score"`
	TotalReward  float64 `json:"total_reward"`
	GamesPlayed  int     `json:"games_played"`
}

type Point struct {
	X, Y int
}

// Color represents RGB values for snake color
type Color struct {
	R, G, B uint8
}

type Snake struct {
	Body              []Point
	Direction         Point
	Score             int
	AI                *ai.QLearning
	LastState         ai.State
	LastAction        ai.Action
	Dead              bool
	Food              Point
	GameOver          bool
	SessionHigh       int
	AllTimeHigh       int
	Scores            []int
	AverageScore      float64
	Mutex             sync.RWMutex // Protect snake state during updates
	game              *Game        // Reference to the game for grid dimensions
	Reproducing       bool         // Whether the snake is currently reproducing
	ReproduceCycles   int          // Number of cycles to stay still during reproduction
	ReproduceWith     *Snake       // The other snake involved in reproduction
	HasReproducedEver bool         // Track if snake has reproduced in its lifetime
	Color             Color        // Snake body color
}

type Game struct {
	UUID              string
	PreviousGameID    string
	Grid              Grid
	Snakes            []*Snake
	AIMode            bool
	StartTime         time.Time
	TotalGames        int
	Stats             GameStats
	FoodCount         int   // Track total amount of food in game
	PopulationHistory []int // Track population over time
	LastPopCheck      int   // Last cycle population was checked
	Iterations        int   // Track number of update cycles
}

func NewGame(width, height int, previousGameID string) *Game {
	gameUUID := uuid.New().String()
	startTime := time.Now()

	game := &Game{
		UUID:           gameUUID,
		PreviousGameID: previousGameID,
		Grid: Grid{
			Width:  width,
			Height: height,
		},
		AIMode:     true,
		StartTime:  startTime,
		TotalGames: 0,
		Stats: GameStats{
			UUID:           gameUUID,
			PreviousGameID: previousGameID,
			StartTime:      startTime,
			AgentStats:     make([]AgentStats, NumAgents),
		},
		FoodCount:         0,
		PopulationHistory: make([]int, 0),
		LastPopCheck:      0,
		Iterations:        0,
	}

	// Initialize snakes in different positions
	game.Snakes = make([]*Snake, NumAgents)

	// Calculate evenly distributed starting positions
	startPositions := make([][2]int, NumAgents)
	for i := 0; i < NumAgents; i++ {
		// Distribute snakes in a grid-like pattern
		row := i / int(math.Sqrt(float64(NumAgents)))
		col := i % int(math.Sqrt(float64(NumAgents)))

		startPositions[i] = [2]int{
			width * (col + 1) / (int(math.Sqrt(float64(NumAgents))) + 1),  // Distribute across width
			height * (row + 1) / (int(math.Sqrt(float64(NumAgents))) + 1), // Distribute across height
		}
	}

	// Create initial agents based on CPU cores
	for i := 0; i < NumAgents; i++ {
		agent := ai.NewQLearning(nil, 0)
		game.Snakes[i] = NewSnake(startPositions[i], agent, game)
		game.FoodCount++ // Count initial food for each snake
	}

	// Create game directory
	os.MkdirAll(filepath.Join("data", "games", gameUUID, "agents"), 0755)

	return game
}

func generateRandomColor() Color {
	return Color{
		R: uint8(rand.Intn(256)),
		G: uint8(rand.Intn(256)),
		B: uint8(rand.Intn(256)),
	}
}

func mutateColor(baseColor Color) Color {
	// Mutate each color component by ±30
	mutateComponent := func(c uint8) uint8 {
		delta := uint8(rand.Intn(61) - 30) // Range: -30 to +30
		if delta > c {
			return delta - c // Handle underflow
		}
		if uint16(c)+uint16(delta) > 255 {
			return 255 // Handle overflow
		}
		return c + delta
	}

	return Color{
		R: mutateComponent(baseColor.R),
		G: mutateComponent(baseColor.G),
		B: mutateComponent(baseColor.B),
	}
}

func NewSnake(startPos [2]int, agent *ai.QLearning, game *Game) *Snake {
	snake := &Snake{
		Body:              []Point{{X: startPos[0], Y: startPos[1]}},
		Direction:         Point{X: 1, Y: 0}, // Start moving right
		AI:                agent,
		Score:             0,
		Dead:              false,
		GameOver:          false,
		Scores:            make([]int, 0),
		AverageScore:      0,
		game:              game,
		Reproducing:       false,
		ReproduceCycles:   0,
		ReproduceWith:     nil,
		HasReproducedEver: false,
		Color:             generateRandomColor(), // Random color for new snake (will be overridden if breeding)
	}
	snake.Food = snake.SpawnFood()
	return snake
}

func (s *Snake) SpawnFood() Point {
	for {
		food := Point{
			X: rand.Intn(s.game.Grid.Width),
			Y: rand.Intn(s.game.Grid.Height),
		}

		// Check if food spawned on current snake
		collision := false
		for _, p := range s.Body {
			if p == food {
				collision = true
				break
			}
		}

		// Only check other snakes if they exist
		if !collision && s.game.Snakes != nil {
			for _, snake := range s.game.Snakes {
				if snake != nil && snake != s { // Skip current snake and nil snakes
					for _, p := range snake.Body {
						if p == food {
							collision = true
							break
						}
					}
					if collision {
						break
					}
				}
			}
		}

		if !collision {
			return food
		}
	}
}

func (s *Snake) GetState() ai.State {
	head := s.Body[len(s.Body)-1]

	// Calculate relative food direction
	foodDir := [2]int{
		sign(s.Food.X - head.X),
		sign(s.Food.Y - head.Y),
	}

	// Calculate Manhattan distance to food
	foodDist := abs(s.Food.X-head.X) + abs(s.Food.Y-head.Y)

	// Check dangers in all directions
	dangers := [4]bool{
		s.IsDanger(Point{X: head.X, Y: head.Y - 1}), // Up
		s.IsDanger(Point{X: head.X + 1, Y: head.Y}), // Right
		s.IsDanger(Point{X: head.X, Y: head.Y + 1}), // Down
		s.IsDanger(Point{X: head.X - 1, Y: head.Y}), // Left
	}

	return ai.NewState(foodDir, foodDist, dangers)
}

func (s *Snake) IsDanger(p Point) bool {
	// Check wall collision
	if p.X < 0 || p.X >= s.game.Grid.Width || p.Y < 0 || p.Y >= s.game.Grid.Height {
		return true
	}

	// Check collision with self
	for _, sp := range s.Body {
		if p == sp {
			return true
		}
	}

	// Check collision with other snakes
	for _, otherSnake := range s.game.Snakes {
		if otherSnake != nil && otherSnake != s { // Skip self and nil snakes
			// Check if point collides with other snake's head
			otherHead := otherSnake.Body[len(otherSnake.Body)-1]
			if p == otherHead {
				// If it's another snake's head, return false to encourage reproduction
				return false
			}

			// Check if point collides with other snake's body (excluding head)
			for i := 0; i < len(otherSnake.Body)-1; i++ {
				if p == otherSnake.Body[i] {
					// If it's another snake's body, treat it like self collision
					return true
				}
			}
		}
	}

	return false
}

func sign(x int) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (s *Snake) Update(g *Game) {
	s.Mutex.RLock()
	if s.GameOver || s.Dead {
		s.Mutex.RUnlock()
		return
	}
	s.Mutex.RUnlock()

	// Handle reproduction state
	if s.Reproducing {
		if s.ReproduceCycles > 0 {
			s.ReproduceCycles--
			return // Stay still during reproduction
		} else {
			// Reproduction complete, move in opposite directions
			if s.ReproduceWith != nil {
				s.Direction = Point{X: 1, Y: 0}                // Move right
				s.ReproduceWith.Direction = Point{X: -1, Y: 0} // Move left

				// Spawn new agent in random position within radius 2
				head := s.Body[len(s.Body)-1]
				for i := 0; i < 10; i++ { // Try up to 10 times to find valid position
					newX := head.X + rand.Intn(5) - 2 // Random position ±2 from head
					newY := head.Y + rand.Intn(5) - 2

					// Check if position is valid
					if newX >= 0 && newX < g.Grid.Width && newY >= 0 && newY < g.Grid.Height {
						newAgent := ai.NewQLearning(nil, 0)
						// Create new snake with mutated color from parent
						newSnake := NewSnake([2]int{newX, newY}, newAgent, g)
						newSnake.Color = mutateColor(s.Color) // Mutate parent's color
						g.Snakes = append(g.Snakes, newSnake)
						g.FoodCount++ // Count food for new snake
						break
					}
				}
			}
			// Reset reproduction state
			s.Reproducing = false
			s.ReproduceWith = nil
			return
		}
	}

	currentState := s.GetState()

	// Get AI action if not first move
	var action ai.Action
	if s.LastState != (ai.State{}) {
		// Update Q-values
		s.AI.Update(s.LastState, s.LastAction, currentState)
	}

	// Get next action
	action = s.AI.GetAction(currentState)
	s.LastState = currentState
	s.LastAction = action

	// Convert action to direction
	switch action {
	case ai.Up:
		if s.Direction.Y != 1 {
			s.Direction = Point{0, -1}
		}
	case ai.Right:
		if s.Direction.X != -1 {
			s.Direction = Point{1, 0}
		}
	case ai.Down:
		if s.Direction.Y != -1 {
			s.Direction = Point{0, 1}
		}
	case ai.Left:
		if s.Direction.X != 1 {
			s.Direction = Point{-1, 0}
		}
	}

	// Calculate new head position
	head := s.Body[len(s.Body)-1]
	newHead := Point{
		X: head.X + s.Direction.X,
		Y: head.Y + s.Direction.Y,
	}

	s.Mutex.Lock()
	// Check collisions with other snakes
	for _, otherSnake := range g.Snakes {
		if otherSnake != nil && otherSnake != s {
			otherHead := otherSnake.Body[len(otherSnake.Body)-1]

			// Check head-to-head collision (reproduction)
			if newHead == otherHead {
				// If either snake has reproduced before, they die
				if s.HasReproducedEver || otherSnake.HasReproducedEver {
					s.Dead = true
					s.GameOver = true
					s.AI.GamesPlayed++
					otherSnake.Dead = true
					otherSnake.GameOver = true
					otherSnake.AI.GamesPlayed++
					s.Mutex.Unlock()
					return
				}

				// Start reproduction process for first-time reproduction
				s.Reproducing = true
				s.ReproduceCycles = 5 // Stay still for 5 cycles
				s.ReproduceWith = otherSnake
				s.HasReproducedEver = true
				otherSnake.Reproducing = true
				otherSnake.ReproduceCycles = 5
				otherSnake.ReproduceWith = s
				otherSnake.HasReproducedEver = true
				s.Mutex.Unlock()
				return
			}

			// Check if we hit other snake's body
			for _, bodyPart := range otherSnake.Body {
				if newHead == bodyPart {
					s.Dead = true
					s.GameOver = true
					s.AI.GamesPlayed++
					s.Mutex.Unlock()
					return
				}
			}
		}
	}

	// Check wall and self collisions
	if newHead.X < 0 || newHead.X >= g.Grid.Width || newHead.Y < 0 || newHead.Y >= g.Grid.Height {
		s.Dead = true
		s.GameOver = true
		s.AI.GamesPlayed++
		s.Mutex.Unlock()
		return
	}

	for _, bodyPart := range s.Body {
		if newHead == bodyPart {
			s.Dead = true
			s.GameOver = true
			s.AI.GamesPlayed++
			s.Mutex.Unlock()
			return
		}
	}

	// Move snake
	s.Body = append(s.Body, newHead)

	// Check food collision
	if newHead == s.Food {
		s.Score++
		if s.Score > s.SessionHigh {
			s.SessionHigh = s.Score
		}
		if s.Score > s.AllTimeHigh {
			s.AllTimeHigh = s.Score
			// Save new high score
			data := make([]byte, 4)
			binary.LittleEndian.PutUint32(data, uint32(s.AllTimeHigh))
			filename := filepath.Join("data", "games", g.UUID, "agents", s.AI.UUID+"_highscore.txt")
			os.WriteFile(filename, data, 0644)
		}
		g.FoodCount-- // Decrement food count when eaten
		s.Food = s.SpawnFood()
	} else {
		// Remove tail if no food was eaten
		s.Body = s.Body[1:]
	}
	s.Mutex.Unlock()
}

// findBestAgent returns the agent with the highest total reward
func (g *Game) findBestAgent() *ai.QLearning {
	var bestAgent *ai.QLearning
	var bestReward float64 = -1000000 // Very low initial value

	for _, snake := range g.Snakes {
		if snake.AI.TotalReward > bestReward {
			bestReward = snake.AI.TotalReward
			bestAgent = snake.AI
		}
	}

	return bestAgent
}

// allAgentsDead checks if all agents in the game are dead
func (g *Game) allAgentsDead() bool {
	for _, snake := range g.Snakes {
		if !snake.GameOver {
			return false
		}
	}
	return true
}

func (g *Game) SaveGameStats() {
	g.Stats.EndTime = time.Now()
	g.Stats.GameDuration = g.Stats.EndTime.Sub(g.Stats.StartTime).Seconds()
	g.Stats.Iterations = g.Iterations

	// Save game stats to JSON file
	statsPath := filepath.Join("data", "games", g.UUID, "game_stats.json")
	statsJson, _ := json.Marshal(g.Stats)
	os.WriteFile(statsPath, statsJson, 0644)
}

func (g *Game) Update() {
	// Update food spawn counter and iteration count
	foodSpawnCounter++
	g.Iterations++

	// Check population growth
	currentPop := len(g.Snakes)
	g.PopulationHistory = append(g.PopulationHistory, currentPop)

	// Keep history within window
	if len(g.PopulationHistory) > PopGrowthWindow {
		g.PopulationHistory = g.PopulationHistory[1:]
	}

	// Check if population exceeds maximum
	if currentPop > MaxPopulation {
		// Terminate all snakes
		for _, snake := range g.Snakes {
			snake.Dead = true
			snake.GameOver = true
			snake.AI.GamesPlayed++
		}
		g.Snakes = g.Snakes[:NumAgents] // Reset to initial population
		return
	}

	// Update snakes sequentially to prevent mutex contention
	for _, snake := range g.Snakes {
		if snake.GameOver {
			g.TotalGames++
		}
		snake.Update(g)
	}

	// Generate new food only if we have less than half the number of agents worth of food
	minFood := len(g.Snakes) / 2
	if foodSpawnCounter >= FoodSpawnCycles && g.FoodCount < minFood {
		for _, snake := range g.Snakes {
			if !snake.Dead {
				snake.Food = snake.SpawnFood()
				g.FoodCount++
			}
		}
		foodSpawnCounter = 0
	}

	// When all agents are dead, save stats and reset them without creating new ones
	if g.allAgentsDead() {
		// Save game stats before resetting
		g.SaveGameStats()

		// Find the best performing agent
		bestAgent := g.findBestAgent()

		// Reset existing agents using the best agent's Q-table
		for _, snake := range g.Snakes {
			snake.Mutex.Lock()

			// Keep the existing agent but update its Q-table with the best agent's
			snake.AI.QTable = bestAgent.QTable.Copy()
			snake.AI.Mutate(0.1) // 10% mutation rate

			// Reset snake position and state but keep scores, agent, reproduction status, and mutate color
			oldScores := snake.Scores
			oldAvgScore := snake.AverageScore
			oldSessionHigh := snake.SessionHigh
			oldAllTimeHigh := snake.AllTimeHigh
			oldAI := snake.AI
			oldMutex := snake.Mutex
			oldHasReproducedEver := snake.HasReproducedEver
			oldColor := snake.Color

			*snake = *NewSnake([2]int{snake.Body[0].X, snake.Body[0].Y}, oldAI, g)
			snake.Mutex = oldMutex
			snake.HasReproducedEver = oldHasReproducedEver
			// Keep same color if breeding or resetting from previous game
			if snake.HasReproducedEver || oldAllTimeHigh > 0 {
				snake.Color = mutateColor(oldColor) // Mutate previous color
			} else {
				snake.Color = generateRandomColor() // Random color for fresh start
			}

			// Restore scores
			snake.Scores = oldScores
			snake.AverageScore = oldAvgScore
			snake.SessionHigh = oldSessionHigh
			snake.AllTimeHigh = oldAllTimeHigh

			snake.Mutex.Unlock()
		}
	} else {
		// When an agent dies, let it stay dead until all agents are dead
		for _, snake := range g.Snakes {
			if snake.GameOver {
				// Save Q-table less frequently
				if snake.AI.GamesPlayed%50 == 0 {
					filename := ai.GetQTableFilename(g.UUID, snake.AI.UUID)
					snake.AI.SaveQTable(filename)
				}
			}
		}
	}
}
