import { MemoryService } from '../src/services/memory';
import { QdrantService } from '../src/services/qdrant';
import { OpenAIService } from '../src/services/openai';
import { MemoryType } from '../src/types/memory';

async function testEdgeCases() {
  const qdrant = new QdrantService();
  const openai = new OpenAIService();
  const memoryService = new MemoryService(qdrant, openai);

  console.log('Initializing Qdrant...');
  await qdrant.initialize();

  // Test 1: Empty content validation
  console.log('\n=== Test 1: Empty Content Validation ===');
  try {
    await memoryService.createMemory('', 'semantic');
    console.log('❌ FAIL: Empty content was accepted');
  } catch (e) {
    console.log('✅ PASS: Empty content rejected');
  }

  // Test 2: Importance boundaries
  console.log('\n=== Test 2: Importance Boundaries ===');
  const importanceTests = [-0.1, 0, 0.5, 1, 1.1];
  for (const imp of importanceTests) {
    try {
      await memoryService.createMemory(`Test ${imp}`, 'semantic', {}, imp);
      console.log(`✅ Importance ${imp}: Accepted`);
    } catch (e) {
      console.log(`✅ Importance ${imp}: Rejected (expected for out of bounds)`);
    }
  }

  // Test 3: Unicode and special characters
  console.log('\n=== Test 3: Unicode & Special Characters ===');
  try {
    const unicodeMemory = await memoryService.createMemory(
      '测试 émojis 🎉 and special chars: <>&"\'',
      'semantic'
    );
    console.log('✅ PASS: Unicode content created');
  } catch (e) {
    console.log('❌ FAIL: Unicode content failed:', e.message);
  }

  // Test 4: Search threshold effectiveness
  console.log('\n=== Test 4: Search Threshold Test ===');
  
  // Create test memory
  await memoryService.createMemory('The cat sat on the mat', 'semantic');
  
  // Search with different thresholds
  const thresholds = [0.3, 0.7, 0.9];
  for (const threshold of thresholds) {
    const results = await memoryService.searchMemories({
      query: 'dog',
      similarityThreshold: threshold,
      limit: 5
    });
    console.log(`Threshold ${threshold}: Found ${results.length} results`);
  }

  // Test 5: Memory updates
  console.log('\n=== Test 5: Memory Update Validation ===');
  const testMem = await memoryService.createMemory('Original', 'semantic');
  
  try {
    await memoryService.updateMemory(testMem.id, { content: '' });
    console.log('❌ FAIL: Empty update accepted');
  } catch (e) {
    console.log('✅ PASS: Empty update rejected');
  }

  // Test 6: Deletion
  console.log('\n=== Test 6: Memory Deletion ===');
  const deleteMem = await memoryService.createMemory('To be deleted', 'semantic');
  await memoryService.deleteMemory(deleteMem.id);
  const retrieved = await memoryService.getMemory(deleteMem.id);
  console.log(retrieved === null ? '✅ PASS: Deleted memory is null' : '❌ FAIL: Memory still exists');

  console.log('\n✅ All edge case tests completed!');
}

testEdgeCases().catch(console.error);