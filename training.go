package main

import (
	"fmt"
	"time"
)

// TrainingPhase rappresenta una fase del curriculum learning
type TrainingPhase struct {
	name         string
	episodes     int
	gridWidth    int
	gridHeight   int
	foodReward   float64
	deathPenalty float64
}

// TrainWithCurriculum implementa il curriculum learning per l'addestramento
func TrainWithCurriculum(totalEpisodes int) {
	phases := []TrainingPhase{
		{
			name:         "Basic Movement",
			episodes:     totalEpisodes / 5,
			gridWidth:    8, // Start with very small grid
			gridHeight:   8,
			foodReward:   50.0,  // Small reward to encourage basic movement
			deathPenalty: -50.0, // Mild penalty to learn boundaries
		},
		{
			name:         "Food Collection",
			episodes:     totalEpisodes / 5,
			gridWidth:    10,
			gridHeight:   10,
			foodReward:   100.0,  // Increased reward for food
			deathPenalty: -100.0, // Balanced penalty
		},
		{
			name:         "Survival Skills",
			episodes:     totalEpisodes / 5,
			gridWidth:    12,
			gridHeight:   12,
			foodReward:   200.0,  // Higher reward for food
			deathPenalty: -150.0, // Increased penalty
		},
		{
			name:         "Advanced Navigation",
			episodes:     totalEpisodes / 5,
			gridWidth:    15,
			gridHeight:   15,
			foodReward:   300.0,  // High reward for food
			deathPenalty: -200.0, // High penalty for mistakes
		},
		{
			name:         "Expert Performance",
			episodes:     totalEpisodes / 5,
			gridWidth:    20,
			gridHeight:   20,
			foodReward:   400.0,  // Maximum reward
			deathPenalty: -250.0, // Maximum penalty
		},
	}

	for phaseIndex, phase := range phases {
		fmt.Printf("\nFase %d: %s\n", phaseIndex+1, phase.name)
		fmt.Printf("Episodi: %d, Griglia: %dx%d\n", phase.episodes, phase.gridWidth, phase.gridHeight)

		// Crea il gioco con le dimensioni della fase corrente
		game := NewGame(phase.gridWidth, phase.gridHeight)
		agent := NewSnakeAgent(game)

		// Imposta i reward specifici per questa fase
		agent.SetRewardValues(phase.foodReward, phase.deathPenalty)

		// Statistiche della fase
		bestScore := 0
		totalScore := 0
		startTime := time.Now()

		for episode := 0; episode < phase.episodes; episode++ {
			// Reset del gioco per il nuovo episodio
			agent.Reset()

			// Esegui l'episodio
			for !game.GetSnake().Dead {
				agent.Update()
			}

			// Aggiorna le statistiche
			score := game.GetSnake().Score
			totalScore += score
			if score > bestScore {
				bestScore = score
			}

			// Periodic logging with more metrics
			if (episode+1)%50 == 0 {
				avgScore := float64(totalScore) / 50.0
				elapsed := time.Since(startTime)
				epsilon := agent.GetEpsilon()
				fmt.Printf("[Phase %s] Episode %d/%d - Avg Score: %.2f, Best Score: %d, Epsilon: %.4f, Time: %s\n",
					phase.name, episode+1, phase.episodes, avgScore, bestScore, epsilon, elapsed.Round(time.Second))

				// Reset statistics for next batch
				totalScore = 0
				startTime = time.Now()
			}

			// Save weights more frequently during training
			if (episode+1)%500 == 0 {
				if err := agent.SaveWeights(); err != nil {
					fmt.Printf("Error saving weights: %v\n", err)
				} else {
					fmt.Printf("Weights saved at episode %d\n", episode+1)
				}
			}
		}
	}

	fmt.Println("\nAddestramento completato!")
}
