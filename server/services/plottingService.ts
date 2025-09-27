import { createCanvas, loadImage } from 'canvas';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { SpaceObject, ConjunctionEvent } from '@shared/schema';

interface PlotConfig {
  width: number;
  height: number;
  backgroundColor: string;
  title: string;
  xLabel: string;
  yLabel: string;
  gridLines: boolean;
  legend: boolean;
}

interface TimeSeriesData {
  timestamp: Date;
  value: number;
  label?: string;
}

interface ScatterData {
  x: number;
  y: number;
  label?: string;
  color?: string;
}

interface BarData {
  label: string;
  value: number;
  color?: string;
}

export class PlottingService {
  private static instance: PlottingService;
  private plotsDir: string;

  public static getInstance(): PlottingService {
    if (!PlottingService.instance) {
      PlottingService.instance = new PlottingService();
    }
    return PlottingService.instance;
  }

  constructor() {
    this.plotsDir = join(process.cwd(), 'server', 'plots');
    this.ensurePlotsDirectory();
  }

  private ensurePlotsDirectory(): void {
    if (!existsSync(this.plotsDir)) {
      mkdirSync(this.plotsDir, { recursive: true });
    }
  }

  // Create time series plot
  async createTimeSeriesPlot(
    data: TimeSeriesData[],
    config: Partial<PlotConfig> = {}
  ): Promise<string> {
    const defaultConfig: PlotConfig = {
      width: 800,
      height: 600,
      backgroundColor: '#1a1a1a',
      title: 'Time Series Plot',
      xLabel: 'Time',
      yLabel: 'Value',
      gridLines: true,
      legend: true,
      ...config,
    };

    const canvas = createCanvas(defaultConfig.width, defaultConfig.height);
    const ctx = canvas.getContext('2d');

    // Set background
    ctx.fillStyle = defaultConfig.backgroundColor;
    ctx.fillRect(0, 0, defaultConfig.width, defaultConfig.height);

    // Set up margins
    const margin = { top: 60, right: 60, bottom: 80, left: 80 };
    const plotWidth = defaultConfig.width - margin.left - margin.right;
    const plotHeight = defaultConfig.height - margin.top - margin.bottom;

    // Find data bounds
    const timestamps = data.map(d => d.timestamp.getTime());
    const values = data.map(d => d.value);
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);

    // Add padding to value range
    const valueRange = maxValue - minValue;
    const paddedMinValue = minValue - valueRange * 0.1;
    const paddedMaxValue = maxValue + valueRange * 0.1;

    // Draw title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(defaultConfig.title, defaultConfig.width / 2, 30);

    // Draw axes labels
    ctx.font = '14px Arial';
    ctx.fillText(defaultConfig.xLabel, defaultConfig.width / 2, defaultConfig.height - 20);
    
    ctx.save();
    ctx.translate(20, defaultConfig.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(defaultConfig.yLabel, 0, 0);
    ctx.restore();

    // Draw grid lines
    if (defaultConfig.gridLines) {
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;
      
      // Vertical grid lines
      for (let i = 0; i <= 10; i++) {
        const x = margin.left + (i / 10) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotHeight);
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = margin.top + (i / 10) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotWidth, y);
        ctx.stroke();
      }
    }

    // Draw axes
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Draw data line
    if (data.length > 1) {
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < data.length; i++) {
        const x = margin.left + ((timestamps[i] - minTime) / (maxTime - minTime)) * plotWidth;
        const y = margin.top + plotHeight - ((values[i] - paddedMinValue) / (paddedMaxValue - paddedMinValue)) * plotHeight;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Draw data points
    ctx.fillStyle = '#00ff00';
    for (let i = 0; i < data.length; i++) {
      const x = margin.left + ((timestamps[i] - minTime) / (maxTime - minTime)) * plotWidth;
      const y = margin.top + plotHeight - ((values[i] - paddedMinValue) / (paddedMaxValue - paddedMinValue)) * plotHeight;
      
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Save plot
    const filename = `timeseries_${Date.now()}.png`;
    const filepath = join(this.plotsDir, filename);
    const buffer = canvas.toBuffer('image/png');
    writeFileSync(filepath, buffer);

    return filepath;
  }

  // Create scatter plot
  async createScatterPlot(
    data: ScatterData[],
    config: Partial<PlotConfig> = {}
  ): Promise<string> {
    const defaultConfig: PlotConfig = {
      width: 800,
      height: 600,
      backgroundColor: '#1a1a1a',
      title: 'Scatter Plot',
      xLabel: 'X',
      yLabel: 'Y',
      gridLines: true,
      legend: false,
      ...config,
    };

    const canvas = createCanvas(defaultConfig.width, defaultConfig.height);
    const ctx = canvas.getContext('2d');

    // Set background
    ctx.fillStyle = defaultConfig.backgroundColor;
    ctx.fillRect(0, 0, defaultConfig.width, defaultConfig.height);

    // Set up margins
    const margin = { top: 60, right: 60, bottom: 80, left: 80 };
    const plotWidth = defaultConfig.width - margin.left - margin.right;
    const plotHeight = defaultConfig.height - margin.top - margin.bottom;

    // Find data bounds
    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);

    // Add padding
    const xRange = maxX - minX;
    const yRange = maxY - minY;
    const paddedMinX = minX - xRange * 0.1;
    const paddedMaxX = maxX + xRange * 0.1;
    const paddedMinY = minY - yRange * 0.1;
    const paddedMaxY = maxY + yRange * 0.1;

    // Draw title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(defaultConfig.title, defaultConfig.width / 2, 30);

    // Draw axes labels
    ctx.font = '14px Arial';
    ctx.fillText(defaultConfig.xLabel, defaultConfig.width / 2, defaultConfig.height - 20);
    
    ctx.save();
    ctx.translate(20, defaultConfig.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(defaultConfig.yLabel, 0, 0);
    ctx.restore();

    // Draw grid lines
    if (defaultConfig.gridLines) {
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;
      
      // Vertical grid lines
      for (let i = 0; i <= 10; i++) {
        const x = margin.left + (i / 10) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotHeight);
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = margin.top + (i / 10) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotWidth, y);
        ctx.stroke();
      }
    }

    // Draw axes
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Draw data points
    for (const point of data) {
      const x = margin.left + ((point.x - paddedMinX) / (paddedMaxX - paddedMinX)) * plotWidth;
      const y = margin.top + plotHeight - ((point.y - paddedMinY) / (paddedMaxY - paddedMinY)) * plotHeight;
      
      ctx.fillStyle = point.color || '#00ff00';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Save plot
    const filename = `scatter_${Date.now()}.png`;
    const filepath = join(this.plotsDir, filename);
    const buffer = canvas.toBuffer('image/png');
    writeFileSync(filepath, buffer);

    return filepath;
  }

  // Create bar chart
  async createBarChart(
    data: BarData[],
    config: Partial<PlotConfig> = {}
  ): Promise<string> {
    const defaultConfig: PlotConfig = {
      width: 800,
      height: 600,
      backgroundColor: '#1a1a1a',
      title: 'Bar Chart',
      xLabel: 'Category',
      yLabel: 'Value',
      gridLines: true,
      legend: false,
      ...config,
    };

    const canvas = createCanvas(defaultConfig.width, defaultConfig.height);
    const ctx = canvas.getContext('2d');

    // Set background
    ctx.fillStyle = defaultConfig.backgroundColor;
    ctx.fillRect(0, 0, defaultConfig.width, defaultConfig.height);

    // Set up margins
    const margin = { top: 60, right: 60, bottom: 80, left: 80 };
    const plotWidth = defaultConfig.width - margin.left - margin.right;
    const plotHeight = defaultConfig.height - margin.top - margin.bottom;

    // Find data bounds
    const values = data.map(d => d.value);
    const maxValue = Math.max(...values);
    const paddedMaxValue = maxValue * 1.1;

    // Draw title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(defaultConfig.title, defaultConfig.width / 2, 30);

    // Draw axes labels
    ctx.font = '14px Arial';
    ctx.fillText(defaultConfig.xLabel, defaultConfig.width / 2, defaultConfig.height - 20);
    
    ctx.save();
    ctx.translate(20, defaultConfig.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(defaultConfig.yLabel, 0, 0);
    ctx.restore();

    // Draw grid lines
    if (defaultConfig.gridLines) {
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;
      
      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = margin.top + (i / 10) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotWidth, y);
        ctx.stroke();
      }
    }

    // Draw axes
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Draw bars
    const barWidth = plotWidth / data.length * 0.8;
    const barSpacing = plotWidth / data.length * 0.2;

    for (let i = 0; i < data.length; i++) {
      const barHeight = (data[i].value / paddedMaxValue) * plotHeight;
      const x = margin.left + i * (barWidth + barSpacing) + barSpacing / 2;
      const y = margin.top + plotHeight - barHeight;

      ctx.fillStyle = data[i].color || '#00ff00';
      ctx.fillRect(x, y, barWidth, barHeight);

      // Draw label
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(data[i].label, x + barWidth / 2, margin.top + plotHeight + 20);
    }

    // Save plot
    const filename = `barchart_${Date.now()}.png`;
    const filepath = join(this.plotsDir, filename);
    const buffer = canvas.toBuffer('image/png');
    writeFileSync(filepath, buffer);

    return filepath;
  }

  // Create orbital distribution plot
  async createOrbitalDistributionPlot(objects: SpaceObject[]): Promise<string> {
    const data = objects.map(obj => ({
      x: obj.altitude || 0,
      y: obj.inclination || 0,
      label: obj.name,
      color: this.getObjectColor(obj),
    }));

    return this.createScatterPlot(data, {
      title: 'Orbital Distribution',
      xLabel: 'Altitude (km)',
      yLabel: 'Inclination (degrees)',
    });
  }

  // Create risk distribution plot
  async createRiskDistributionPlot(objects: SpaceObject[]): Promise<string> {
    const riskCounts = {
      low: objects.filter(obj => obj.riskLevel === 'low').length,
      medium: objects.filter(obj => obj.riskLevel === 'medium').length,
      high: objects.filter(obj => obj.riskLevel === 'high').length,
      critical: objects.filter(obj => obj.riskLevel === 'critical').length,
    };

    const data: BarData[] = [
      { label: 'Low', value: riskCounts.low, color: '#00ff00' },
      { label: 'Medium', value: riskCounts.medium, color: '#ffff00' },
      { label: 'High', value: riskCounts.high, color: '#ff8800' },
      { label: 'Critical', value: riskCounts.critical, color: '#ff0000' },
    ];

    return this.createBarChart(data, {
      title: 'Risk Level Distribution',
      xLabel: 'Risk Level',
      yLabel: 'Number of Objects',
    });
  }

  // Get color for object based on type and risk
  private getObjectColor(obj: SpaceObject): string {
    if (obj.riskLevel === 'critical') return '#ff0000';
    if (obj.riskLevel === 'high') return '#ff8800';
    if (obj.riskLevel === 'medium') return '#ffff00';
    if (obj.type === 'satellite') return '#00ff00';
    if (obj.type === 'debris') return '#888888';
    return '#0088ff';
  }

  // Get list of generated plots
  getPlotsList(): string[] {
    // This would typically read from the plots directory
    return [];
  }

  // Delete a plot file
  deletePlot(filename: string): boolean {
    try {
      const filepath = join(this.plotsDir, filename);
      if (existsSync(filepath)) {
        writeFileSync(filepath, ''); // Clear file
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error deleting plot:', error);
      return false;
    }
  }
}

export const plottingService = PlottingService.getInstance();
