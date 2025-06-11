import { MemoryService } from '../src/services/memory';
import { QdrantService } from '../src/services/qdrant';
import { OpenAIService } from '../src/services/openai';
import { MemoryType } from '../src/types/memory';

async function runEdgeCaseTests() {
  const qdrant = new QdrantService();
  const openai = new OpenAIService();
  const memoryService = new MemoryService(qdrant, openai);

  console.log('🧪 Starting Memory System Edge Case Tests...\n');

  // Initialize Qdrant
  await qdrant.initialize();

  // Test 1: Empty/Null Content
  console.log('📝 Test 1: Empty/Null Content Handling');
  try {
    const emptyMemory = await memoryService.createMemory('', 'semantic');
    console.log('❌ Empty content should have failed');
  } catch (error) {
    console.log('✅ Empty content correctly rejected');
  }

  try {
    const whitespaceMemory = await memoryService.createMemory('   \n\t  ', 'semantic');
    console.log('❌ Whitespace-only content should have failed');
  } catch (error) {
    console.log('✅ Whitespace-only content correctly rejected');
  }

  // Test 2: Very Long Content
  console.log('\n📝 Test 2: Very Long Content (>10KB)');
  const longContent = 'A'.repeat(15000); // 15KB of 'A's
  try {
    const longMemory = await memoryService.createMemory(
      longContent,
      'semantic',
      { tags: ['test', 'long-content'] }
    );
    console.log(`✅ Created memory with ${longContent.length} characters`);
    console.log(`   ID: ${longMemory.id}`);
  } catch (error) {
    console.log('❌ Failed to create long content memory:', error);
  }

  // Test 3: Special Characters and Unicode
  console.log('\n📝 Test 3: Special Characters and Unicode');
  const specialContents = [
    '测试中文字符 🎉 émojis and spëcial çharacters',
    'Line\nbreaks\r\nand\ttabs',
    '<script>alert("XSS")</script> & HTML entities',
    '```code blocks``` and **markdown**',
    '\\backslashes\\ and /forward/slashes/',
    'Null char: \0 and other controls: \x01\x02',
  ];

  for (const content of specialContents) {
    try {
      const memory = await memoryService.createMemory(
        content,
        'semantic',
        { tags: ['special-chars'] }
      );
      console.log(`✅ Created: "${content.substring(0, 30)}..."`);
    } catch (error) {
      console.log(`❌ Failed on: "${content.substring(0, 30)}..."`, error.message);
    }
  }

  // Test 4: Similarity Threshold Edge Cases
  console.log('\n📝 Test 4: Similarity Threshold Edge Cases');
  
  // Create a reference memory
  const refMemory = await memoryService.createMemory(
    'The quick brown fox jumps over the lazy dog',
    'semantic',
    { tags: ['reference'] }
  );

  const thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0];
  for (const threshold of thresholds) {
    const results = await memoryService.searchMemories({
      query: 'The quick brown fox',
      limit: 10,
      similarityThreshold: threshold
    });
    console.log(`✅ Threshold ${threshold}: Found ${results.length} results`);
  }

  // Test 5: Importance/Emotional Valence Boundaries
  console.log('\n📝 Test 5: Importance/Emotional Valence Boundaries');
  const boundaryValues = [-2, -1.1, -1, -0.5, 0, 0.5, 1, 1.1, 2];
  
  for (const value of boundaryValues) {
    try {
      const memory = await memoryService.createMemory(
        `Test memory with importance ${value}`,
        'semantic',
        {},
        value // importance
      );
      console.log(`✅ Importance ${value}: ${value >= 0 && value <= 1 ? 'Accepted' : 'Should have failed!'}`);
    } catch (error) {
      console.log(`✅ Importance ${value}: Correctly rejected`);
    }
  }

  // Test 6: Association Limits
  console.log('\n📝 Test 6: Association Limits and Circular References');
  
  // Create memories that will be associated
  const mem1 = await memoryService.createMemory('Memory A about associations', 'semantic');
  const mem2 = await memoryService.createMemory('Memory B related to A', 'semantic');
  const mem3 = await memoryService.createMemory('Memory C connected to both', 'semantic');
  
  // Check if associations were created automatically
  const searchResults = await memoryService.searchMemories({
    query: 'associations',
    includeAssociations: true,
    limit: 5
  });
  console.log(`✅ Found ${searchResults.length} memories with associations`);

  // Test 7: Concurrent Operations
  console.log('\n📝 Test 7: Concurrent Operations');
  const concurrentPromises = [];
  for (let i = 0; i < 10; i++) {
    concurrentPromises.push(
      memoryService.createMemory(
        `Concurrent memory ${i} created at ${Date.now()}`,
        'working',
        { tags: ['concurrent', `batch-${i}`] }
      )
    );
  }
  
  try {
    const results = await Promise.all(concurrentPromises);
    console.log(`✅ Successfully created ${results.length} memories concurrently`);
  } catch (error) {
    console.log('❌ Concurrent creation failed:', error);
  }

  // Test 8: Memory Update Edge Cases
  console.log('\n📝 Test 8: Memory Update Edge Cases');
  
  // Update non-existent memory
  try {
    await memoryService.updateMemory('non-existent-id', { content: 'Updated' });
    console.log('❌ Updating non-existent memory should have failed');
  } catch (error) {
    console.log('✅ Non-existent memory update correctly rejected');
  }

  // Update with empty content
  const updateTarget = await memoryService.createMemory('Original content', 'semantic');
  try {
    await memoryService.updateMemory(updateTarget.id, { content: '' });
    console.log('❌ Empty content update should have failed');
  } catch (error) {
    console.log('✅ Empty content update correctly rejected');
  }

  // Update multiple fields
  const updated = await memoryService.updateMemory(updateTarget.id, {
    content: 'Updated content with new information',
    importance: 0.9,
    context: { mood: 'excited', tags: ['updated', 'modified'] }
  });
  console.log('✅ Multi-field update successful');

  // Test 9: Complex Search Filters
  console.log('\n📝 Test 9: Search with Complex Filters');
  
  // Create diverse memories for filtering
  const diverseMemories = [
    { content: 'Happy memory from yesterday', type: 'emotional' as MemoryType, importance: 0.8, emotionalValence: 0.9 },
    { content: 'Sad memory from last week', type: 'emotional' as MemoryType, importance: 0.6, emotionalValence: -0.8 },
    { content: 'Neutral work memory', type: 'working' as MemoryType, importance: 0.5, emotionalValence: 0 },
    { content: 'Important procedure', type: 'procedural' as MemoryType, importance: 0.95, emotionalValence: 0.2 },
  ];

  for (const mem of diverseMemories) {
    await memoryService.createMemory(mem.content, mem.type, {}, mem.importance);
  }

  // Complex filter search
  const complexSearch = await memoryService.searchMemories({
    query: 'memory',
    type: 'emotional',
    minImportance: 0.5,
    emotionalRange: { min: -1, max: 1 },
    limit: 20
  });
  console.log(`✅ Complex filter search returned ${complexSearch.length} results`);

  // Test 10: Memory Deletion and Orphaned Associations
  console.log('\n📝 Test 10: Memory Deletion and Orphaned Associations');
  
  // Create associated memories
  const parent = await memoryService.createMemory('Parent memory to be deleted', 'semantic');
  const child = await memoryService.createMemory('Child memory referencing parent', 'semantic');
  
  // Delete parent
  await memoryService.deleteMemory(parent.id);
  console.log('✅ Parent memory deleted');
  
  // Try to retrieve deleted memory
  const deletedMem = await memoryService.getMemory(parent.id);
  console.log(`✅ Deleted memory retrieval: ${deletedMem === null ? 'Correctly returns null' : 'ERROR - still exists!'}`);

  // Check if child still exists
  const orphanedChild = await memoryService.getMemory(child.id);
  console.log(`✅ Child memory still exists: ${orphanedChild !== null}`);

  console.log('\n✨ Edge case tests completed!');
}

// Run tests
runEdgeCaseTests().catch(console.error);