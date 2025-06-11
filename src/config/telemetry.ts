import { MeterProvider } from '@opentelemetry/sdk-metrics';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';
import { metrics } from '@opentelemetry/api';

export function initializeTelemetry() {
  // Create Prometheus exporter with /metrics endpoint
  const prometheusExporter = new PrometheusExporter({
    port: parseInt(process.env.METRICS_PORT || '9090'),
    endpoint: '/metrics',
  }, () => {
    console.log('Prometheus metrics server started on port', process.env.METRICS_PORT || '9090');
  });

  // Create meter provider
  const meterProvider = new MeterProvider({
    readers: [prometheusExporter],
  });

  // Set global meter provider
  metrics.setGlobalMeterProvider(meterProvider);

  return {
    meterProvider,
    prometheusExporter,
  };
}

// Create meters for custom metrics
export const meter = metrics.getMeter('mcp-memory-server', '1.0.0');

// Initialize observable metrics
export function setupObservableMetrics() {
  // These will be updated by the application
  let memoryItemsCount = 0;
  let memoryCollectionsCount = 1; // Default collection
  
  // Set up observable callbacks
  const memoryItemsGauge = meter.createObservableGauge('mcp_memory_items_total', {
    description: 'Total number of items in memory storage',
    unit: '1',
  });
  
  const memoryCollectionsGauge = meter.createObservableGauge('mcp_memory_collections_total', {
    description: 'Total number of memory collections',
    unit: '1',
  });
  
  meter.addBatchObservableCallback((observableResult) => {
    observableResult.observe(memoryItemsGauge, memoryItemsCount);
    observableResult.observe(memoryCollectionsGauge, memoryCollectionsCount);
  }, [memoryItemsGauge, memoryCollectionsGauge]);
  
  return {
    updateMemoryItemsCount: (count: number) => { memoryItemsCount = count; },
    updateMemoryCollectionsCount: (count: number) => { memoryCollectionsCount = count; },
  };
}