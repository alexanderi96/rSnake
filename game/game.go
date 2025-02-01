package game

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"snake-game/ai"
	"sync"
	"time"

	"github.com/google/uuid"
)

var NumAgents = runtime.NumCPU() // Number of agents equals number of CPU cores

// Grid dimensions are now part of Game struct to be dynamic
type Grid struct {
	Width  int
	Height int
}

type GameStats struct {
	UUID       string       `json:"uuid"`
	StartTime  time.Time    `json:"start_time"`
	EndTime    time.Time    `json:"end_time"`
	AgentStats []AgentStats `json:"agent_stats"`
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

type Snake struct {
	Body         []Point
	Direction    Point
	Score        int
	AI           *ai.QLearning
	LastState    ai.State
	LastAction   ai.Action
	Dead         bool
	Food         Point
	GameOver     bool
	SessionHigh  int
	AllTimeHigh  int
	Scores       []int
	AverageScore float64
	Mutex        sync.RWMutex // Protect snake state during updates
	game         *Game        // Reference to the game for grid dimensions
}

type Game struct {
	UUID       string
	Grid       Grid
	Snakes     []*Snake
	AIMode     bool
	StartTime  time.Time
	TotalGames int
	Stats      GameStats
}

func NewGame(width, height int) *Game {
	gameUUID := uuid.New().String()
	startTime := time.Now()

	game := &Game{
		UUID: gameUUID,
		Grid: Grid{
			Width:  width,
			Height: height,
		},
		AIMode:     true,
		StartTime:  startTime,
		TotalGames: 0,
		Stats: GameStats{
			UUID:       gameUUID,
			StartTime:  startTime,
			AgentStats: make([]AgentStats, NumAgents),
		},
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
	}

	// Create game directory
	os.MkdirAll(filepath.Join("data", "games", gameUUID, "agents"), 0755)

	return game
}

func NewSnake(startPos [2]int, agent *ai.QLearning, game *Game) *Snake {
	snake := &Snake{
		Body:         []Point{{X: startPos[0], Y: startPos[1]}},
		Direction:    Point{X: 1, Y: 0}, // Start moving right
		AI:           agent,
		Score:        0,
		Dead:         false,
		GameOver:     false,
		Scores:       make([]int, 0),
		AverageScore: 0,
		game:         game,
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

		// Check if food spawned on snake
		collision := false
		for _, p := range s.Body {
			if p == food {
				collision = true
				break
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
	// Check collisions
	if s.IsDanger(newHead) {
		s.Dead = true
		s.GameOver = true
		s.AI.GamesPlayed++

		// Update scores
		if len(s.Scores) >= 50 { // maxScores constant
			s.Scores = s.Scores[1:]
		}
		s.Scores = append(s.Scores, s.Score)

		// Calculate average score
		sum := 0
		for _, score := range s.Scores {
			sum += score
		}
		s.AverageScore = float64(sum) / float64(len(s.Scores))

		// Save Q-table solo ogni 10 partite
		if s.AI.GamesPlayed%10 == 0 {
			filename := ai.GetQTableFilename(g.UUID, s.AI.UUID)
			s.AI.SaveQTable(filename)
		}

		s.Mutex.Unlock()
		return
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

func (g *Game) Update() {
	// Update snakes sequentially to prevent mutex contention
	for _, snake := range g.Snakes {
		if snake.GameOver {
			g.TotalGames++
		}
		snake.Update(g)
	}

	// Check if all agents are dead
	if g.allAgentsDead() {
		// Find the best performing agent
		bestAgent := g.findBestAgent()

		// Create new agents using the best agent's Q-table
		for i := range g.Snakes {
			// Create a new agent with the best agent's Q-table and some mutation
			newAgent := ai.NewQLearning(bestAgent.QTable, 0.1) // 10% mutation rate

			snake := g.Snakes[i]
			snake.Mutex.Lock()

			// Reset snake but keep scores
			oldScores := snake.Scores
			oldAvgScore := snake.AverageScore
			oldSessionHigh := snake.SessionHigh
			oldAllTimeHigh := snake.AllTimeHigh
			oldMutex := snake.Mutex

			// Create new snake with the new agent
			*snake = *NewSnake([2]int{snake.Body[0].X, snake.Body[0].Y}, newAgent, g)
			snake.Mutex = oldMutex

			// Restore scores
			snake.Scores = oldScores
			snake.AverageScore = oldAvgScore
			snake.SessionHigh = oldSessionHigh
			snake.AllTimeHigh = oldAllTimeHigh
			snake.Mutex.Unlock()

			// Save Q-table of the new agent
			filename := ai.GetQTableFilename(g.UUID, newAgent.UUID)
			newAgent.SaveQTable(filename)
		}
	} else {
		// Handle individual dead snakes normally
		for _, snake := range g.Snakes {
			if snake.GameOver {
				snake.Mutex.Lock()
				// Reset snake but keep scores
				oldScores := snake.Scores
				oldAvgScore := snake.AverageScore
				oldSessionHigh := snake.SessionHigh
				oldAllTimeHigh := snake.AllTimeHigh
				oldMutex := snake.Mutex // Keep the mutex

				*snake = *NewSnake([2]int{snake.Body[0].X, snake.Body[0].Y}, snake.AI, g)
				snake.Mutex = oldMutex // Restore the mutex

				snake.Scores = oldScores
				snake.AverageScore = oldAvgScore
				snake.SessionHigh = oldSessionHigh
				snake.AllTimeHigh = oldAllTimeHigh
				snake.Mutex.Unlock()

				// Save Q-table less frequently and synchronously
				if snake.AI.GamesPlayed%50 == 0 {
					filename := ai.GetQTableFilename(g.UUID, snake.AI.UUID)
					snake.AI.SaveQTable(filename)
				}
			}
		}
	}
}
