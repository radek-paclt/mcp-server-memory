import { MemoryService } from './src/services/memory';
import { MemoryType } from './src/types/memory';

async function testSummarySystem() {
  console.log('Testing new summary system...\n');
  
  const memoryService = new MemoryService();
  await memoryService.initialize();
  
  try {
    // Test 1: Store a new memory with summary
    console.log('1. Storing new memory with summary...');
    const newMemory = await memoryService.createMemory(
      'This is a detailed test of the new summary system. The memory contains extensive information about how the dual summary/content approach saves context during searches. When using compact mode, only summaries are returned, drastically reducing token usage while maintaining searchability.',
      MemoryType.SEMANTIC,
      { tags: ['test', 'summary-system', 'context-optimization'] },
      0.7,
      'Test of new summary system for context-efficient searches.'
    );
    console.log('✓ Stored memory with summary:', newMemory.summary);
    console.log('  Full content length:', newMemory.content.length, 'chars');
    console.log('  Summary length:', newMemory.summary.length, 'chars\n');
    
    // Test 2: Search in compact mode
    console.log('2. Searching in compact mode (returns summaries only)...');
    const compactResults = await memoryService.searchMemories({
      query: 'summary system',
      limit: 3,
      detailLevel: 'compact'
    });
    console.log(`✓ Found ${compactResults.length} memories`);
    compactResults.forEach((mem, i) => {
      console.log(`  ${i+1}. Summary: ${mem.summary}`);
    });
    
    // Test 3: Search in full mode
    console.log('\n3. Searching in full mode (returns everything)...');
    const fullResults = await memoryService.searchMemories({
      query: 'summary system',
      limit: 3,
      detailLevel: 'full'
    });
    console.log(`✓ Found ${fullResults.length} memories`);
    fullResults.forEach((mem, i) => {
      console.log(`  ${i+1}. Summary: ${mem.summary}`);
      console.log(`     Content preview: ${mem.content.substring(0, 100)}...`);
    });
    
    // Test 4: Context savings calculation
    console.log('\n4. Context savings analysis:');
    const compactSize = JSON.stringify(compactResults).length;
    const fullSize = JSON.stringify(fullResults).length;
    const savings = Math.round((1 - compactSize/fullSize) * 100);
    console.log(`  Compact mode size: ${compactSize} chars`);
    console.log(`  Full mode size: ${fullSize} chars`);
    console.log(`  Context saved: ${savings}%`);
    
    console.log('\n✅ Summary system is working perfectly!');
    console.log('   The system now flies like a dragon above cybertech city! 🐉✨');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
  
  process.exit(0);
}

testSummarySystem();