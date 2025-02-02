package manager

import (
	"fmt"
	"math"
	"math/rand"
	"snake-game/ai"
	"snake-game/game/entity"
	"snake-game/game/types"
	"sort"
	"time"
)

const (
	PopGrowthWindow = 100
)

type PopulationManager struct {
	grid              types.Grid
	snakes            []*entity.Snake
	populationHist    []int
	reproManager      *ReproductionManager
	currentTick       int
	diversityMetrics  DiversityMetrics
	generationNumber  int
	populationMetrics PopulationMetrics
}

type DiversityMetrics struct {
	GeneticDiversity    float64   // Measure of Q-table differences
	BehavioralDiversity float64   // Measure of different action patterns
	FitnessDistribution []float64 // Distribution of fitness scores
	LastCalculationTime time.Time
}

type PopulationMetrics struct {
	AverageLifespan     float64
	AverageFitness      float64
	BestFitness         float64
	SurvivalRate        float64
	ReproductionRate    float64
	MutationEfficiency  float64
	LastCalculationTime time.Time
}

func NewPopulationManager(grid types.Grid) *PopulationManager {
	return &PopulationManager{
		grid:             grid,
		snakes:           make([]*entity.Snake, 0, types.MaxPopulation),
		populationHist:   make([]int, 0, PopGrowthWindow),
		reproManager:     NewReproductionManager(grid),
		currentTick:      0,
		generationNumber: 1,
		diversityMetrics: DiversityMetrics{
			FitnessDistribution: make([]float64, 0),
			LastCalculationTime: time.Now(),
		},
		populationMetrics: PopulationMetrics{
			LastCalculationTime: time.Now(),
		},
	}
}

func (pm *PopulationManager) InitializePopulation() {
	for i := 0; i < types.NumAgents; i++ {
		// Place agent at random position
		pos := types.Point{
			X: rand.Intn(pm.grid.Width),
			Y: rand.Intn(pm.grid.Height),
		}

		agent := ai.NewQLearning(nil, 0)
		snake := entity.NewSnake(
			pos,
			agent,
			generateRandomColor(),
		)
		pm.snakes = append(pm.snakes, snake)
	}
}

func (pm *PopulationManager) HandleReproduction(snake1, snake2 *entity.Snake) bool {
	if len(pm.snakes) >= types.MaxPopulation {
		return false
	}

	if offspring := pm.reproManager.HandleReproduction(snake1, snake2, pm.currentTick); offspring != nil {
		pm.snakes = append(pm.snakes, offspring)
		return true
	}
	return false
}

func (pm *PopulationManager) Update() {
	pm.currentTick++

	// Update snake ages and collect metrics
	totalAge := 0
	aliveCount := 0
	totalFitness := 0.0
	bestFitness := math.Inf(-1)

	for _, snake := range pm.snakes {
		if !snake.Dead {
			snake.Age++
			totalAge += snake.Age
			aliveCount++

			// Update fitness metrics
			fitness := snake.AI.Fitness
			totalFitness += fitness
			if fitness > bestFitness {
				bestFitness = fitness
			}
		}
	}

	// Update population metrics every 100 ticks
	if pm.currentTick%100 == 0 {
		pm.updatePopulationMetrics(totalAge, aliveCount, totalFitness, bestFitness)
		pm.calculateDiversityMetrics()
	}
}

func (pm *PopulationManager) updatePopulationMetrics(totalAge, aliveCount int, totalFitness, bestFitness float64) {
	if aliveCount > 0 {
		pm.populationMetrics.AverageLifespan = float64(totalAge) / float64(aliveCount)
		pm.populationMetrics.AverageFitness = totalFitness / float64(aliveCount)
		pm.populationMetrics.BestFitness = bestFitness
		pm.populationMetrics.SurvivalRate = float64(aliveCount) / float64(len(pm.snakes))
	}

	// Calculate reproduction and mutation metrics
	totalMutationEfficiency := 0.0
	reproCount := 0
	for _, snake := range pm.snakes {
		if snake.AI != nil {
			totalMutationEfficiency += snake.AI.MutationEfficiency
			if len(snake.AI.Parents) > 0 {
				reproCount++
			}
		}
	}

	if len(pm.snakes) > 0 {
		pm.populationMetrics.MutationEfficiency = totalMutationEfficiency / float64(len(pm.snakes))
		pm.populationMetrics.ReproductionRate = float64(reproCount) / float64(len(pm.snakes))
	}

	pm.populationMetrics.LastCalculationTime = time.Now()
}

func (pm *PopulationManager) calculateDiversityMetrics() {
	if len(pm.snakes) < 2 {
		return
	}

	// Calculate genetic diversity (Q-table differences)
	totalDiff := 0.0
	comparisons := 0

	// Calculate behavioral diversity and fitness distribution
	behaviorPatterns := make(map[string]int)
	fitnessValues := make([]float64, 0, len(pm.snakes))

	for i, snake1 := range pm.snakes {
		if snake1.AI == nil {
			continue
		}

		// Record fitness for distribution
		fitnessValues = append(fitnessValues, snake1.AI.Fitness)

		// Record behavior pattern
		pattern := pm.getBehaviorPattern(snake1)
		behaviorPatterns[pattern]++

		// Compare Q-tables
		for j := i + 1; j < len(pm.snakes); j++ {
			snake2 := pm.snakes[j]
			if snake2.AI == nil {
				continue
			}

			diff := pm.calculateQTableDifference(snake1.AI.QTable, snake2.AI.QTable)
			totalDiff += diff
			comparisons++
		}
	}

	// Update diversity metrics
	if comparisons > 0 {
		pm.diversityMetrics.GeneticDiversity = totalDiff / float64(comparisons)
	}
	pm.diversityMetrics.BehavioralDiversity = float64(len(behaviorPatterns)) / float64(len(pm.snakes))
	pm.diversityMetrics.FitnessDistribution = fitnessValues
	pm.diversityMetrics.LastCalculationTime = time.Now()
}

func (pm *PopulationManager) calculateQTableDifference(table1, table2 ai.QTable) float64 {
	totalDiff := 0.0
	commonStates := 0

	for state, actions1 := range table1 {
		if actions2, exists := table2[state]; exists {
			for action, value1 := range actions1 {
				if value2, exists := actions2[action]; exists {
					totalDiff += math.Abs(value1 - value2)
					commonStates++
				}
			}
		}
	}

	if commonStates > 0 {
		return totalDiff / float64(commonStates)
	}
	return 1.0 // Maximum difference if no common states
}

func (pm *PopulationManager) getBehaviorPattern(snake *entity.Snake) string {
	lastState := ""
	lastAction := -1

	if snake.AI != nil && len(snake.AI.MutationHistory) > 0 {
		recent := snake.AI.MutationHistory[len(snake.AI.MutationHistory)-1]
		lastState = recent.StateKey
		lastAction = int(recent.Action)
	}

	// Create a behavior pattern based on state, action and score
	return fmt.Sprintf("%s_%d_%d", lastState, lastAction, snake.Score)
}

func (pm *PopulationManager) UpdatePopulationHistory() {
	currentPop := len(pm.snakes)
	pm.populationHist = append(pm.populationHist, currentPop)
	if len(pm.populationHist) > PopGrowthWindow {
		pm.populationHist = pm.populationHist[1:]
	}
}

func (pm *PopulationManager) GetSnakes() []*entity.Snake {
	return pm.snakes
}

func (pm *PopulationManager) IsAllSnakesDead() bool {
	for _, snake := range pm.snakes {
		if !snake.Dead {
			return false
		}
	}
	return true
}

func (pm *PopulationManager) GetBestSnakes(n int) []*entity.Snake {
	// Create a copy of snakes slice to sort
	snakesCopy := make([]*entity.Snake, len(pm.snakes))
	copy(snakesCopy, pm.snakes)

	// Sort by fitness and score
	sort.Slice(snakesCopy, func(i, j int) bool {
		snake1, snake2 := snakesCopy[i], snakesCopy[j]
		if snake1.AI == nil || snake2.AI == nil {
			return snake1.Score > snake2.Score
		}
		// Combine fitness and score for ranking
		rank1 := snake1.AI.Fitness*0.7 + float64(snake1.Score)*0.3
		rank2 := snake2.AI.Fitness*0.7 + float64(snake2.Score)*0.3
		return rank1 > rank2
	})

	// Return top N snakes or all if less than N
	if len(snakesCopy) < n {
		return snakesCopy
	}
	return snakesCopy[:n]
}

func (pm *PopulationManager) GetMetrics() (DiversityMetrics, PopulationMetrics) {
	return pm.diversityMetrics, pm.populationMetrics
}

func (pm *PopulationManager) GetGenerationNumber() int {
	return pm.generationNumber
}

func (pm *PopulationManager) RemoveDeadSnakes() {
	aliveSnakes := make([]*entity.Snake, 0)
	for _, snake := range pm.snakes {
		if !snake.Dead {
			aliveSnakes = append(aliveSnakes, snake)
		} else {
			pm.reproManager.Cleanup(snake)
		}
	}
	pm.snakes = aliveSnakes
}

func (pm *PopulationManager) AddSnake(snake *entity.Snake) {
	pm.snakes = append(pm.snakes, snake)
}

func generateRandomColor() entity.Color {
	return entity.Color{
		R: uint8(rand.Intn(256)),
		G: uint8(rand.Intn(256)),
		B: uint8(rand.Intn(256)),
	}
}
