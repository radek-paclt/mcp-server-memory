# Similarity Threshold Analysis for Memory Search

## Test Results Summary

Based on testing with various queries and memories, here are the findings for different similarity thresholds:

### Threshold Behavior

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.2 | Very permissive - returns many loosely related results | Exploratory searches, brainstorming |
| 0.3 | Permissive - good balance of recall | **Recommended default** for general use |
| 0.4 | Moderate - returns reasonably related content | Focused searches |
| 0.5 | Strict - only closely matching content | Precise searches |
| 0.6+ | Very strict - may miss relevant results | Exact or near-exact matches only |

### Query-Specific Results

1. **Specific technical queries** (e.g., "customer analytics XGBoost"):
   - Work well even with higher thresholds (0.5-0.6)
   - Direct matches score highly

2. **General concept queries** (e.g., "programming", "debugging"):
   - Need lower thresholds (0.2-0.4)
   - Many related memories at varying similarity levels

3. **Emotional/contextual queries** (e.g., "frustration with bugs"):
   - Require lower thresholds (0.3-0.4)
   - Semantic similarity varies based on emotional content

## Recommendations

1. **Default Setting**: Use `SIMILARITY_THRESHOLD=0.3` in the environment configuration
   - Provides good balance between precision and recall
   - Captures related concepts without too much noise

2. **Dynamic Thresholds**: Consider implementing query-type detection:
   - Technical/specific queries: 0.4-0.5
   - General/exploratory queries: 0.2-0.3
   - Exact searches: 0.6+

3. **User Control**: Allow users to adjust threshold per search if needed:
   - Add optional parameter in search API
   - Provide presets like "strict", "balanced", "broad"

## Implementation Notes

The current implementation uses cosine similarity with OpenAI's text-embedding-3-small model. Similarity scores typically range:
- 0.7-1.0: Very similar or identical content
- 0.5-0.7: Closely related content
- 0.3-0.5: Related concepts
- 0.2-0.3: Loosely related
- <0.2: Mostly unrelated

The threshold of 0.3 captures most relevant results while filtering out noise.