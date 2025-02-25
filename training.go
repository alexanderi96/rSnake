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
			name:         "Sopravvivenza",
			episodes:     totalEpisodes / 3,
			gridWidth:    10, // Griglia più piccola per iniziare
			gridHeight:   10,
			foodReward:   100.0,  // Reward moderato per il cibo
			deathPenalty: -500.0, // Penalità severa per la morte
		},
		{
			name:         "Raccolta Cibo",
			episodes:     totalEpisodes / 3,
			gridWidth:    15, // Griglia media
			gridHeight:   15,
			foodReward:   300.0,  // Reward maggiore per il cibo
			deathPenalty: -250.0, // Penalità moderata
		},
		{
			name:         "Prestazioni Complete",
			episodes:     totalEpisodes / 3,
			gridWidth:    20, // Griglia completa
			gridHeight:   20,
			foodReward:   500.0,  // Reward massimo per il cibo
			deathPenalty: -250.0, // Mantiene penalità moderata
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

			// Log periodico
			if (episode+1)%100 == 0 {
				avgScore := float64(totalScore) / 100.0
				elapsed := time.Since(startTime)
				fmt.Printf("Episodio %d/%d - Score Medio: %.2f, Miglior Score: %d, Tempo: %s\n",
					episode+1, phase.episodes, avgScore, bestScore, elapsed.Round(time.Second))

				// Reset delle statistiche per il prossimo batch
				totalScore = 0
				startTime = time.Now()
			}

			// Salva i pesi periodicamente
			if (episode+1)%1000 == 0 {
				if err := agent.SaveWeights(); err != nil {
					fmt.Printf("Errore nel salvataggio dei pesi: %v\n", err)
				} else {
					fmt.Printf("Pesi salvati all'episodio %d\n", episode+1)
				}
			}
		}
	}

	fmt.Println("\nAddestramento completato!")
}
