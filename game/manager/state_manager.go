package manager

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"snake-game/game/entity"
	"snake-game/game/types"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

type GameStats struct {
	UUID             string       `json:"uuid"`
	PreviousGameID   string       `json:"previous_game_id"`
	SessionStartTime time.Time    `json:"session_start_time"`
	GameStartTime    time.Time    `json:"game_start_time"`
	GameEndTime      time.Time    `json:"game_end_time"`
	GameDuration     float64      `json:"game_duration"`
	TotalSessionTime float64      `json:"total_session_time"`
	RoundsPlayed     int          `json:"rounds_played"`
	CurrentRound     int          `json:"current_round"`
	Iterations       int          `json:"iterations"`
	AgentStats       []AgentStats `json:"agent_stats"`
	PopulationStats  PopStats     `json:"population_stats"`
	BackupPath       string       `json:"backup_path,omitempty"`
	LastBackupTime   time.Time    `json:"last_backup_time,omitempty"`
}

type PopStats struct {
	PopulationSize      int     `json:"population_size"`
	AverageLifespan     float64 `json:"average_lifespan"`
	PopulationDiversity float64 `json:"population_diversity"`
	BestFitness         float64 `json:"best_fitness"`
	AverageFitness      float64 `json:"average_fitness"`
	GenerationNumber    int     `json:"generation_number"`
}

type AgentStats struct {
	UUID               string    `json:"uuid"`
	Score              int       `json:"score"`
	AverageScore       float64   `json:"average_score"`
	TotalReward        float64   `json:"total_reward"`
	RoundsPlayed       int       `json:"rounds_played"`
	MutationEfficiency float64   `json:"mutation_efficiency"`
	LastMutationTime   time.Time `json:"last_mutation_time,omitempty"`
	LearningRate       float64   `json:"learning_rate"`
	Fitness            float64   `json:"fitness"`
	Generation         int       `json:"generation"`
	Parents            []string  `json:"parents,omitempty"`
}

type StateManager struct {
	UUID           string
	PreviousGameID string
	grid           types.Grid
	stats          GameStats
	iterations     int
	sessionStart   time.Time
	gameStartTime  time.Time
	roundsPlayed   int
	collisionMgr   *CollisionManager
	popManager     *PopulationManager
	foodManager    *FoodManager
	statsBuffer    []GameStats
	backupManager  *BackupManager
}

type BackupManager struct {
	backupDir    string
	maxBackups   int
	compressData bool
	mutex        sync.RWMutex
}

func NewBackupManager(dir string, maxBackups int, compress bool) *BackupManager {
	return &BackupManager{
		backupDir:    dir,
		maxBackups:   maxBackups,
		compressData: compress,
		mutex:        sync.RWMutex{},
	}
}

func NewStateManager(grid types.Grid, previousGameID string, collisionMgr *CollisionManager, popManager *PopulationManager) *StateManager {
	backupDir := filepath.Join("data", "backups")
	os.MkdirAll(backupDir, 0755)
	gameUUID := uuid.New().String()
	now := time.Now()

	// Ensure data directories exist
	os.MkdirAll(filepath.Join("data", "games"), 0755)
	os.MkdirAll(filepath.Join("data", "games", gameUUID, "agents"), 0755)

	sm := &StateManager{
		UUID:           gameUUID,
		PreviousGameID: previousGameID,
		grid:           grid,
		stats: GameStats{
			UUID:             gameUUID,
			PreviousGameID:   previousGameID,
			SessionStartTime: now,
			GameStartTime:    now,
			RoundsPlayed:     0,
			CurrentRound:     1,
			AgentStats:       make([]AgentStats, 0),
			PopulationStats:  PopStats{GenerationNumber: 1},
		},
		iterations:    0,
		sessionStart:  now,
		gameStartTime: now,
		roundsPlayed:  0,
		collisionMgr:  collisionMgr,
		popManager:    popManager,
		foodManager:   NewFoodManager(grid, collisionMgr),
		statsBuffer:   make([]GameStats, 0, 100),
		backupManager: NewBackupManager(backupDir, 5, true),
	}

	return sm
}

func (sm *StateManager) Update(snakes []*entity.Snake) {
	sm.iterations++

	// Check if all snakes are dead to reset game start time
	allDead := true
	for _, snake := range snakes {
		if snake == nil {
			continue
		}
		snake.Mutex.RLock()
		isDead := snake.Dead
		snake.Mutex.RUnlock()
		if !isDead {
			allDead = false
			break
		}
	}
	if allDead {
		sm.gameStartTime = time.Now()
	}

	// Update food manager
	sm.foodManager.Update(snakes)

	// Update stats for each snake
	for _, snake := range snakes {
		if snake == nil {
			continue
		}

		snake.Mutex.RLock()
		isDead := snake.Dead
		if isDead {
			snake.Mutex.RUnlock()
			continue
		}

		// Copy values while holding lock
		score := snake.Score
		avgScore := snake.AverageScore
		aiUUID := snake.AI.UUID
		totalReward := snake.AI.TotalReward
		gamesPlayed := snake.AI.GamesPlayed
		snake.Mutex.RUnlock()

		// Update agent stats
		found := false
		for i, stats := range sm.stats.AgentStats {
			if stats.UUID == aiUUID {
				sm.stats.AgentStats[i].Score = score
				sm.stats.AgentStats[i].AverageScore = avgScore
				sm.stats.AgentStats[i].TotalReward = totalReward
				sm.stats.AgentStats[i].RoundsPlayed = gamesPlayed
				found = true
				break
			}
		}

		if !found {
			sm.stats.AgentStats = append(sm.stats.AgentStats, AgentStats{
				UUID:         aiUUID,
				Score:        score,
				AverageScore: avgScore,
				TotalReward:  totalReward,
				RoundsPlayed: gamesPlayed,
			})
		}
	}
}

func (sm *StateManager) GetFoodList() []types.Point {
	return sm.foodManager.GetFoodList()
}

func (sm *StateManager) AddFood(food types.Point) {
	sm.foodManager.AddFood(food)
}

func (sm *StateManager) RemoveFood(food types.Point) {
	sm.foodManager.RemoveFood(food)
}

func (sm *StateManager) GenerateFood(snakes []*entity.Snake) types.Point {
	return sm.foodManager.GenerateFood(snakes)
}

type GlobalStats struct {
	TrainingStartTime time.Time `json:"training_start_time"`
	TotalGamesStarted int       `json:"total_games_started"`
	LastGameID        string    `json:"last_game_id"`
	LastUpdateTime    time.Time `json:"last_update_time"`
}

func (sm *StateManager) createBackup() error {
	sm.backupManager.mutex.Lock()
	defer sm.backupManager.mutex.Unlock()

	backupFile := filepath.Join(sm.backupManager.backupDir,
		fmt.Sprintf("backup_%s_%s.json", sm.UUID, time.Now().Format("20060102_150405")))

	data, err := json.MarshalIndent(sm.stats, "", "  ")
	if err != nil {
		return err
	}

	if sm.backupManager.compressData {
		var buf bytes.Buffer
		gz := gzip.NewWriter(&buf)
		if _, err := gz.Write(data); err != nil {
			return err
		}
		gz.Close()
		data = buf.Bytes()
		backupFile += ".gz"
	}

	if err := os.WriteFile(backupFile, data, 0644); err != nil {
		return err
	}

	// Cleanup old backups
	files, _ := filepath.Glob(filepath.Join(sm.backupManager.backupDir, "backup_*.json*"))
	if len(files) > sm.backupManager.maxBackups {
		sort.Strings(files)
		for _, f := range files[:len(files)-sm.backupManager.maxBackups] {
			os.Remove(f)
		}
	}

	sm.stats.BackupPath = backupFile
	sm.stats.LastBackupTime = time.Now()
	return nil
}

func (sm *StateManager) SaveGameStats() {
	now := time.Now()

	// Update game round stats
	sm.stats.GameEndTime = now
	sm.stats.GameDuration = now.Sub(sm.gameStartTime).Seconds()
	sm.stats.TotalSessionTime = now.Sub(sm.sessionStart).Seconds()
	sm.stats.Iterations = sm.iterations
	sm.stats.RoundsPlayed++
	sm.stats.CurrentRound++

	// Create backup before saving
	if err := sm.createBackup(); err != nil {
		log.Printf("Failed to create backup: %v", err)
	}

	// Buffer the stats
	sm.statsBuffer = append(sm.statsBuffer, sm.stats)

	// Save when buffer reaches capacity or on significant events
	if len(sm.statsBuffer) >= 100 || sm.stats.CurrentRound%50 == 0 {
		statsPath := filepath.Join("data", "games", sm.UUID, "game_stats.json")
		statsJson, _ := json.MarshalIndent(sm.statsBuffer, "", "  ")
		if err := os.WriteFile(statsPath, statsJson, 0644); err != nil {
			log.Printf("Failed to save game stats: %v", err)
			return
		}
		// Clear buffer after successful save
		sm.statsBuffer = sm.statsBuffer[:0]
	}

	// Update global stats with error handling
	globalStatsPath := filepath.Join("data", "global_stats.json")
	var globalStats GlobalStats
	var backupPath string

	// Backup existing global stats before updating
	if _, err := os.Stat(globalStatsPath); err == nil {
		backupPath = globalStatsPath + ".bak"
		if err := os.Rename(globalStatsPath, backupPath); err != nil {
			log.Printf("Failed to create global stats backup: %v", err)
			return
		}
	}

	// Try to read existing global stats
	if data, err := os.ReadFile(backupPath); err == nil {
		if err := json.Unmarshal(data, &globalStats); err != nil {
			log.Printf("Failed to parse global stats: %v", err)
			// Restore from backup
			os.Rename(backupPath, globalStatsPath)
			return
		}
	} else {
		// Initialize new global stats if file doesn't exist
		globalStats = GlobalStats{
			TrainingStartTime: sm.sessionStart,
			TotalGamesStarted: 0,
		}
	}

	// Update global stats
	globalStats.TotalGamesStarted++
	globalStats.LastGameID = sm.UUID
	globalStats.LastUpdateTime = now

	// Save updated global stats with validation
	globalStatsJson, err := json.MarshalIndent(globalStats, "", "  ")
	if err != nil {
		log.Printf("Failed to marshal global stats: %v", err)
		if backupPath != "" {
			os.Rename(backupPath, globalStatsPath)
		}
		return
	}

	if err := os.WriteFile(globalStatsPath, globalStatsJson, 0644); err != nil {
		log.Printf("Failed to save global stats: %v", err)
		if backupPath != "" {
			os.Rename(backupPath, globalStatsPath)
		}
		return
	}

	// Remove backup after successful save
	if backupPath != "" {
		os.Remove(backupPath)
	}
}

func (sm *StateManager) GetStats() (GameStats, time.Time) {
	return sm.stats, sm.gameStartTime
}

func (sm *StateManager) GetPopulationManager() *PopulationManager {
	return sm.popManager
}
