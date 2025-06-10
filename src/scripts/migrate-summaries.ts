import { MemoryService } from '../services/memory';
import { QdrantService } from '../services/qdrant';

async function migrateSummaries() {
  console.log('Starting summary migration...');
  
  const memoryService = new MemoryService();
  const qdrantService = new QdrantService();
  
  await memoryService.initialize();
  
  try {
    // Get all memories using search with empty query
    const allMemories = await qdrantService.searchByFilters({
      must: []
    }, 1000); // Get up to 1000 memories
    
    console.log(`Found ${allMemories.length} memories to migrate`);
    
    let migrated = 0;
    for (const memory of allMemories) {
      if (!memory.summary || memory.summary === '') {
        // Generate summary for existing memory
        const summary = memory.content.length > 150 
          ? memory.content.substring(0, 147) + '...'
          : memory.content;
        
        // Update memory with summary
        await qdrantService.updateMemoryMetadata(memory.id, {
          summary: summary
        });
        
        migrated++;
        if (migrated % 10 === 0) {
          console.log(`Migrated ${migrated} memories...`);
        }
      }
    }
    
    console.log(`Migration complete! Added summaries to ${migrated} memories.`);
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
  
  process.exit(0);
}

migrateSummaries();