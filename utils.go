package main

// abs restituisce il valore assoluto di un intero.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// manhattanDistance calcola la distanza di Manhattan tra due punti considerando il wrapping della griglia
func manhattanDistance(p1, p2 Point, width, height int) int {
	dx := abs(p2.X - p1.X)
	dy := abs(p2.Y - p1.Y)

	// Considera il wrapping della griglia
	if dx > width/2 {
		dx = width - dx
	}
	if dy > height/2 {
		dy = height - dy
	}

	return dx + dy
}
