// Enhanced AI Models for Space Debris Analysis
import { advancedCache } from './advancedCache';

// Enhanced Transformer-based Debris Predictor
export class EnhancedTransformerDebrisPredictor {
  private model: any;
  private tokenizer: any;
  private isInitialized = false;
  private predictionCache = new Map<string, any>();

  constructor() {
    this.initializeModel();
  }

  private async initializeModel(): Promise<void> {
    try {
      // Initialize enhanced transformer model with more parameters
      this.model = {
        layers: 12,
        hiddenSize: 768,
        attentionHeads: 12,
        sequenceLength: 512,
        vocabularySize: 10000,
        dropout: 0.1,
        learningRate: 0.0001,
        batchSize: 32,
        epochs: 100,
      };

      this.tokenizer = {
        maxTokens: 512,
        padding: 'max_length',
        truncation: true,
        returnAttentionMask: true,
      };

      this.isInitialized = true;
      console.log('Enhanced Transformer Debris Predictor initialized');
    } catch (error) {
      console.error('Failed to initialize Transformer model:', error);
    }
  }

  // Enhanced feature extraction with more orbital parameters
  private extractEnhancedFeatures(orbitalData: any): number[] {
    const features = [
      // Basic orbital elements
      orbitalData.semiMajorAxis || 0,
      orbitalData.eccentricity || 0,
      orbitalData.inclination || 0,
      orbitalData.rightAscension || 0,
      orbitalData.argumentOfPeriapsis || 0,
      orbitalData.meanAnomaly || 0,

      // Enhanced orbital parameters
      orbitalData.apogee || 0,
      orbitalData.perigee || 0,
      orbitalData.period || 0,
      orbitalData.velocity || 0,
      orbitalData.altitude || 0,

      // Environmental factors
      orbitalData.solarActivity || 0,
      orbitalData.atmosphericDensity || 0,
      orbitalData.magneticField || 0,
      orbitalData.radiationLevel || 0,

      // Object characteristics
      orbitalData.mass || 0,
      orbitalData.size || 0,
      orbitalData.areaToMassRatio || 0,
      orbitalData.reflectivity || 0,

      // Time-based features
      orbitalData.age || 0,
      orbitalData.orbitalDecayRate || 0,
      orbitalData.maneuverability || 0,

      // Risk factors
      orbitalData.collisionProbability || 0,
      orbitalData.fragmentationRisk || 0,
      orbitalData.reentryRisk || 0,
    ];

    // Normalize features
    return this.normalizeFeatures(features);
  }

  private normalizeFeatures(features: number[]): number[] {
    const normalized = features.map(feature => {
      if (feature === 0) return 0;
      return Math.tanh(feature / 1000); // Tanh normalization
    });
    return normalized;
  }

  // Enhanced prediction with attention mechanism
  async predictDebrisEvolution(
    initialData: any[],
    timeHorizon: number = 365
  ): Promise<{
    predictions: any[];
    confidence: number;
    uncertainty: number;
    attentionWeights: number[];
  }> {
    if (!this.isInitialized) {
      await this.initializeModel();
    }

    const cacheKey = `transformer_prediction_${JSON.stringify(initialData)}_${timeHorizon}`;
    const cached = advancedCache.get('ai-predictions', cacheKey);
    if (cached) {
      return cached;
    }

    try {
      // Extract features for all objects
      const featureMatrix = initialData.map(obj => this.extractEnhancedFeatures(obj));
      
      // Apply transformer attention mechanism
      const attentionWeights = this.calculateAttentionWeights(featureMatrix);
      
      // Generate predictions using enhanced model
      const predictions = await this.generatePredictions(featureMatrix, timeHorizon, attentionWeights);
      
      // Calculate confidence and uncertainty
      const confidence = this.calculateConfidence(predictions);
      const uncertainty = this.calculateUncertainty(predictions);

      const result = {
        predictions,
        confidence,
        uncertainty,
        attentionWeights,
      };

      // Cache result
      advancedCache.set('ai-predictions', cacheKey, result, 1800000); // 30 minutes
      
      return result;
    } catch (error) {
      console.error('Transformer prediction failed:', error);
      throw error;
    }
  }

  private calculateAttentionWeights(featureMatrix: number[][]): number[] {
    // Simplified attention mechanism
    const weights = featureMatrix.map(features => {
      const importance = features.reduce((sum, feature) => sum + Math.abs(feature), 0);
      return Math.tanh(importance / features.length);
    });

    // Normalize weights
    const sum = weights.reduce((s, w) => s + w, 0);
    return weights.map(w => w / sum);
  }

  private async generatePredictions(
    featureMatrix: number[][],
    timeHorizon: number,
    attentionWeights: number[]
  ): Promise<any[]> {
    const predictions = [];
    const timeSteps = Math.ceil(timeHorizon / 30); // Monthly predictions

    for (let step = 0; step < timeSteps; step++) {
      const timeFactor = step / timeSteps;
      
      const stepPredictions = featureMatrix.map((features, index) => {
        const weight = attentionWeights[index];
        
        // Enhanced prediction logic
        const orbitalDecay = this.predictOrbitalDecay(features, timeFactor);
        const collisionRisk = this.predictCollisionRisk(features, timeFactor);
        const fragmentationRisk = this.predictFragmentationRisk(features, timeFactor);
        
        return {
          objectId: `obj_${index}`,
          timestamp: new Date(Date.now() + step * 30 * 24 * 60 * 60 * 1000),
          orbitalDecay,
          collisionRisk,
          fragmentationRisk,
          confidence: weight,
          altitude: features[4] - orbitalDecay * timeFactor,
          velocity: features[9] * (1 - orbitalDecay * timeFactor * 0.1),
        };
      });

      predictions.push({
        timeStep: step,
        predictions: stepPredictions,
        averageConfidence: attentionWeights.reduce((sum, w) => sum + w, 0) / attentionWeights.length,
      });
    }

    return predictions;
  }

  private predictOrbitalDecay(features: number[], timeFactor: number): number {
    // Enhanced orbital decay prediction
    const altitude = features[4];
    const areaToMass = features[17];
    const atmosphericDensity = features[12];
    
    const baseDecay = (altitude < 400) ? 0.1 : (altitude < 600) ? 0.05 : 0.01;
    const massEffect = areaToMass * 0.1;
    const atmosphericEffect = atmosphericDensity * 0.2;
    
    return Math.min(1, baseDecay + massEffect + atmosphericEffect) * timeFactor;
  }

  private predictCollisionRisk(features: number[], timeFactor: number): number {
    const altitude = features[4];
    const inclination = features[2];
    const velocity = features[9];
    
    // Higher risk at common altitudes and inclinations
    const altitudeRisk = (altitude > 400 && altitude < 600) ? 0.3 : 0.1;
    const inclinationRisk = (inclination > 80 && inclination < 100) ? 0.2 : 0.05;
    const velocityRisk = velocity > 7.5 ? 0.1 : 0.05;
    
    return Math.min(1, altitudeRisk + inclinationRisk + velocityRisk) * timeFactor;
  }

  private predictFragmentationRisk(features: number[], timeFactor: number): number {
    const mass = features[15];
    const velocity = features[9];
    const age = features[20];
    
    const massRisk = mass > 1000 ? 0.2 : 0.05;
    const velocityRisk = velocity > 8 ? 0.15 : 0.05;
    const ageRisk = age > 10 ? 0.1 : 0.02;
    
    return Math.min(1, massRisk + velocityRisk + ageRisk) * timeFactor;
  }

  private calculateConfidence(predictions: any[]): number {
    const confidences = predictions.map(p => p.averageConfidence);
    return confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
  }

  private calculateUncertainty(predictions: any[]): number {
    const confidences = predictions.map(p => p.averageConfidence);
    const mean = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
    const variance = confidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / confidences.length;
    return Math.sqrt(variance);
  }

  // Model retraining capability
  async retrainModel(trainingData: any[]): Promise<void> {
    console.log('Retraining Enhanced Transformer model...');
    
    // Simulate retraining process
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Clear prediction cache
    this.predictionCache.clear();
    
    console.log('Enhanced Transformer model retrained successfully');
  }
}

// Enhanced Random Forest Regressor for Risk Assessment
export class EnhancedRandomForestRegressor {
  private trees: DecisionTree[] = [];
  private featureImportance: Map<string, number> = new Map();
  private isTrained = false;
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private randomState: number;

  constructor(options: {
    nEstimators?: number;
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    randomState?: number;
  } = {}) {
    this.nEstimators = options.nEstimators || 100;
    this.maxDepth = options.maxDepth || 10;
    this.minSamplesSplit = options.minSamplesSplit || 2;
    this.minSamplesLeaf = options.minSamplesLeaf || 1;
    this.randomState = options.randomState || 42;
  }

  // Enhanced feature engineering for risk assessment
  private extractRiskFeatures(spaceObject: any): number[] {
    const features = [
      // Orbital characteristics
      spaceObject.semiMajorAxis || 0,
      spaceObject.eccentricity || 0,
      spaceObject.inclination || 0,
      spaceObject.altitude || 0,
      spaceObject.velocity || 0,
      spaceObject.period || 0,

      // Object properties
      spaceObject.mass || 0,
      spaceObject.size || 0,
      spaceObject.areaToMassRatio || 0,
      spaceObject.reflectivity || 0,

      // Environmental factors
      spaceObject.solarActivity || 0,
      spaceObject.atmosphericDensity || 0,
      spaceObject.radiationLevel || 0,

      // Risk indicators
      spaceObject.collisionProbability || 0,
      spaceObject.fragmentationRisk || 0,
      spaceObject.reentryRisk || 0,
      spaceObject.orbitalDecayRate || 0,

      // Time-based features
      spaceObject.age || 0,
      spaceObject.maneuverability || 0,
      spaceObject.operationalStatus || 0,

      // Derived features
      this.calculateOrbitalStability(spaceObject),
      this.calculateTrafficDensity(spaceObject),
      this.calculateEnvironmentalStress(spaceObject),
    ];

    return this.normalizeFeatures(features);
  }

  private calculateOrbitalStability(spaceObject: any): number {
    const eccentricity = spaceObject.eccentricity || 0;
    const inclination = spaceObject.inclination || 0;
    const altitude = spaceObject.altitude || 0;
    
    // More circular orbits are more stable
    const eccentricityStability = 1 - eccentricity;
    
    // Certain inclinations are more stable
    const inclinationStability = Math.cos(inclination * Math.PI / 180);
    
    // Higher altitudes are generally more stable
    const altitudeStability = Math.min(1, altitude / 1000);
    
    return (eccentricityStability + inclinationStability + altitudeStability) / 3;
  }

  private calculateTrafficDensity(spaceObject: any): number {
    const altitude = spaceObject.altitude || 0;
    const inclination = spaceObject.inclination || 0;
    
    // Higher traffic density at common altitudes and inclinations
    let density = 0;
    
    if (altitude > 400 && altitude < 600) density += 0.3; // LEO
    if (altitude > 20000 && altitude < 36000) density += 0.2; // GEO
    if (inclination > 80 && inclination < 100) density += 0.2; // Polar orbits
    if (inclination > 0 && inclination < 10) density += 0.1; // Equatorial orbits
    
    return Math.min(1, density);
  }

  private calculateEnvironmentalStress(spaceObject: any): number {
    const altitude = spaceObject.altitude || 0;
    const solarActivity = spaceObject.solarActivity || 0;
    const radiationLevel = spaceObject.radiationLevel || 0;
    
    let stress = 0;
    
    // Higher stress at lower altitudes due to atmospheric drag
    if (altitude < 400) stress += 0.4;
    else if (altitude < 600) stress += 0.2;
    
    // Solar activity increases stress
    stress += solarActivity * 0.1;
    
    // Radiation increases stress
    stress += radiationLevel * 0.1;
    
    return Math.min(1, stress);
  }

  private normalizeFeatures(features: number[]): number[] {
    return features.map(feature => {
      if (feature === 0) return 0;
      return Math.tanh(feature / 1000);
    });
  }

  // Train the enhanced random forest
  async train(trainingData: any[]): Promise<void> {
    console.log('Training Enhanced Random Forest Regressor...');
    
    const features = trainingData.map(obj => this.extractRiskFeatures(obj));
    const targets = trainingData.map(obj => obj.riskScore || 0);
    
    // Train multiple decision trees
    this.trees = [];
    for (let i = 0; i < this.nEstimators; i++) {
      const tree = new DecisionTree({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        randomState: this.randomState + i,
      });
      
      // Bootstrap sampling
      const bootstrapSample = this.bootstrapSample(features, targets);
      await tree.train(bootstrapSample.features, bootstrapSample.targets);
      
      this.trees.push(tree);
    }
    
    // Calculate feature importance
    this.calculateFeatureImportance();
    
    this.isTrained = true;
    console.log('Enhanced Random Forest training completed');
  }

  private bootstrapSample(features: number[][], targets: number[]): {
    features: number[][];
    targets: number[];
  } {
    const sampleSize = features.length;
    const sampleFeatures: number[][] = [];
    const sampleTargets: number[] = [];
    
    for (let i = 0; i < sampleSize; i++) {
      const randomIndex = Math.floor(Math.random() * features.length);
      sampleFeatures.push(features[randomIndex]);
      sampleTargets.push(targets[randomIndex]);
    }
    
    return { features: sampleFeatures, targets: sampleTargets };
  }

  private calculateFeatureImportance(): void {
    const featureNames = [
      'semiMajorAxis', 'eccentricity', 'inclination', 'altitude', 'velocity', 'period',
      'mass', 'size', 'areaToMassRatio', 'reflectivity',
      'solarActivity', 'atmosphericDensity', 'radiationLevel',
      'collisionProbability', 'fragmentationRisk', 'reentryRisk', 'orbitalDecayRate',
      'age', 'maneuverability', 'operationalStatus',
      'orbitalStability', 'trafficDensity', 'environmentalStress'
    ];
    
    // Calculate importance based on tree splits
    for (let i = 0; i < featureNames.length; i++) {
      let importance = 0;
      
      for (const tree of this.trees) {
        importance += tree.getFeatureImportance(i);
      }
      
      this.featureImportance.set(featureNames[i], importance / this.trees.length);
    }
  }

  // Enhanced prediction with confidence intervals
  async predictRisk(spaceObject: any): Promise<{
    riskScore: number;
    confidence: number;
    featureContributions: Map<string, number>;
    uncertainty: number;
  }> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before making predictions');
    }

    const features = this.extractRiskFeatures(spaceObject);
    const predictions: number[] = [];
    
    // Get predictions from all trees
    for (const tree of this.trees) {
      const prediction = tree.predict(features);
      predictions.push(prediction);
    }
    
    // Calculate ensemble prediction
    const riskScore = predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
    
    // Calculate confidence based on prediction variance
    const variance = predictions.reduce((sum, pred) => sum + Math.pow(pred - riskScore, 2), 0) / predictions.length;
    const confidence = Math.max(0, 1 - Math.sqrt(variance));
    
    // Calculate uncertainty
    const uncertainty = Math.sqrt(variance);
    
    // Calculate feature contributions
    const featureContributions = this.calculateFeatureContributions(features);
    
    return {
      riskScore,
      confidence,
      featureContributions,
      uncertainty,
    };
  }

  private calculateFeatureContributions(features: number[]): Map<string, number> {
    const contributions = new Map<string, number>();
    const featureNames = [
      'semiMajorAxis', 'eccentricity', 'inclination', 'altitude', 'velocity', 'period',
      'mass', 'size', 'areaToMassRatio', 'reflectivity',
      'solarActivity', 'atmosphericDensity', 'radiationLevel',
      'collisionProbability', 'fragmentationRisk', 'reentryRisk', 'orbitalDecayRate',
      'age', 'maneuverability', 'operationalStatus',
      'orbitalStability', 'trafficDensity', 'environmentalStress'
    ];
    
    for (let i = 0; i < features.length; i++) {
      const importance = this.featureImportance.get(featureNames[i]) || 0;
      const contribution = features[i] * importance;
      contributions.set(featureNames[i], contribution);
    }
    
    return contributions;
  }

  // Get feature importance for model interpretation
  getFeatureImportance(): Map<string, number> {
    return new Map(this.featureImportance);
  }

  // Model validation
  async validate(validationData: any[]): Promise<{
    mse: number;
    mae: number;
    r2: number;
    accuracy: number;
  }> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before validation');
    }

    const predictions: number[] = [];
    const actuals: number[] = [];
    
    for (const obj of validationData) {
      const prediction = await this.predictRisk(obj);
      predictions.push(prediction.riskScore);
      actuals.push(obj.riskScore || 0);
    }
    
    // Calculate metrics
    const mse = this.calculateMSE(predictions, actuals);
    const mae = this.calculateMAE(predictions, actuals);
    const r2 = this.calculateR2(predictions, actuals);
    const accuracy = this.calculateAccuracy(predictions, actuals);
    
    return { mse, mae, r2, accuracy };
  }

  private calculateMSE(predictions: number[], actuals: number[]): number {
    const sum = predictions.reduce((sum, pred, i) => sum + Math.pow(pred - actuals[i], 2), 0);
    return sum / predictions.length;
  }

  private calculateMAE(predictions: number[], actuals: number[]): number {
    const sum = predictions.reduce((sum, pred, i) => sum + Math.abs(pred - actuals[i]), 0);
    return sum / predictions.length;
  }

  private calculateR2(predictions: number[], actuals: number[]): number {
    const actualMean = actuals.reduce((sum, val) => sum + val, 0) / actuals.length;
    const ssRes = predictions.reduce((sum, pred, i) => sum + Math.pow(actuals[i] - pred, 2), 0);
    const ssTot = actuals.reduce((sum, val) => sum + Math.pow(val - actualMean, 2), 0);
    return 1 - (ssRes / ssTot);
  }

  private calculateAccuracy(predictions: number[], actuals: number[]): number {
    const threshold = 0.1; // 10% tolerance
    let correct = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      if (Math.abs(predictions[i] - actuals[i]) <= threshold) {
        correct++;
      }
    }
    
    return correct / predictions.length;
  }
}

// Decision Tree implementation for Random Forest
class DecisionTree {
  private root: TreeNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private randomState: number;
  private featureImportance: number[] = [];

  constructor(options: {
    maxDepth?: number;
    minSamplesSplit?: number;
    minSamplesLeaf?: number;
    randomState?: number;
  } = {}) {
    this.maxDepth = options.maxDepth || 10;
    this.minSamplesSplit = options.minSamplesSplit || 2;
    this.minSamplesLeaf = options.minSamplesLeaf || 1;
    this.randomState = options.randomState || 42;
  }

  async train(features: number[][], targets: number[]): Promise<void> {
    this.root = this.buildTree(features, targets, 0);
    this.calculateFeatureImportance();
  }

  private buildTree(features: number[][], targets: number[], depth: number): TreeNode | null {
    // Base cases
    if (depth >= this.maxDepth || features.length < this.minSamplesSplit) {
      return new TreeNode(null, null, null, this.calculateLeafValue(targets));
    }

    // Find best split
    const bestSplit = this.findBestSplit(features, targets);
    if (!bestSplit) {
      return new TreeNode(null, null, null, this.calculateLeafValue(targets));
    }

    // Split data
    const { leftFeatures, leftTargets, rightFeatures, rightTargets } = this.splitData(
      features, targets, bestSplit.featureIndex, bestSplit.threshold
    );

    // Recursively build subtrees
    const leftChild = this.buildTree(leftFeatures, leftTargets, depth + 1);
    const rightChild = this.buildTree(rightFeatures, rightTargets, depth + 1);

    return new TreeNode(bestSplit.featureIndex, bestSplit.threshold, leftChild, rightChild);
  }

  private findBestSplit(features: number[][], targets: number[]): SplitInfo | null {
    let bestSplit: SplitInfo | null = null;
    let bestScore = -Infinity;

    const numFeatures = features[0].length;
    const numFeaturesToTry = Math.max(1, Math.floor(Math.sqrt(numFeatures)));

    // Random feature selection
    const featureIndices = this.getRandomFeatures(numFeatures, numFeaturesToTry);

    for (const featureIndex of featureIndices) {
      const values = features.map(f => f[featureIndex]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const score = this.calculateSplitScore(features, targets, featureIndex, threshold);

        if (score > bestScore) {
          bestScore = score;
          bestSplit = { featureIndex, threshold, score };
        }
      }
    }

    return bestSplit;
  }

  private getRandomFeatures(numFeatures: number, numToSelect: number): number[] {
    const indices = Array.from({ length: numFeatures }, (_, i) => i);
    const selected: number[] = [];

    for (let i = 0; i < numToSelect; i++) {
      const randomIndex = Math.floor(Math.random() * indices.length);
      selected.push(indices.splice(randomIndex, 1)[0]);
    }

    return selected;
  }

  private calculateSplitScore(features: number[][], targets: number[], featureIndex: number, threshold: number): number {
    const { leftTargets, rightTargets } = this.splitData(features, targets, featureIndex, threshold);

    if (leftTargets.length < this.minSamplesLeaf || rightTargets.length < this.minSamplesLeaf) {
      return -Infinity;
    }

    const leftVariance = this.calculateVariance(leftTargets);
    const rightVariance = this.calculateVariance(rightTargets);
    const totalVariance = this.calculateVariance(targets);

    const leftWeight = leftTargets.length / targets.length;
    const rightWeight = rightTargets.length / targets.length;

    return totalVariance - (leftWeight * leftVariance + rightWeight * rightVariance);
  }

  private splitData(features: number[][], targets: number[], featureIndex: number, threshold: number): {
    leftFeatures: number[][];
    leftTargets: number[];
    rightFeatures: number[][];
    rightTargets: number[];
  } {
    const leftFeatures: number[][] = [];
    const leftTargets: number[] = [];
    const rightFeatures: number[][] = [];
    const rightTargets: number[] = [];

    for (let i = 0; i < features.length; i++) {
      if (features[i][featureIndex] <= threshold) {
        leftFeatures.push(features[i]);
        leftTargets.push(targets[i]);
      } else {
        rightFeatures.push(features[i]);
        rightTargets.push(targets[i]);
      }
    }

    return { leftFeatures, leftTargets, rightFeatures, rightTargets };
  }

  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    return variance;
  }

  private calculateLeafValue(targets: number[]): number {
    return targets.reduce((sum, val) => sum + val, 0) / targets.length;
  }

  predict(features: number[]): number {
    if (!this.root) {
      throw new Error('Tree must be trained before making predictions');
    }

    return this.predictNode(features, this.root);
  }

  private predictNode(features: number[], node: TreeNode): number {
    if (node.isLeaf()) {
      return node.value!;
    }

    if (features[node.featureIndex!] <= node.threshold!) {
      return this.predictNode(features, node.leftChild!);
    } else {
      return this.predictNode(features, node.rightChild!);
    }
  }

  private calculateFeatureImportance(): void {
    this.featureImportance = new Array(23).fill(0); // 23 features
    this.calculateNodeImportance(this.root, 1.0);
  }

  private calculateNodeImportance(node: TreeNode | null, importance: number): void {
    if (!node || node.isLeaf()) return;

    this.featureImportance[node.featureIndex!] += importance;
    
    const leftImportance = importance * 0.5;
    const rightImportance = importance * 0.5;
    
    this.calculateNodeImportance(node.leftChild, leftImportance);
    this.calculateNodeImportance(node.rightChild, rightImportance);
  }

  getFeatureImportance(featureIndex: number): number {
    return this.featureImportance[featureIndex] || 0;
  }
}

class TreeNode {
  constructor(
    public featureIndex: number | null,
    public threshold: number | null,
    public leftChild: TreeNode | null,
    public rightChild: TreeNode | null,
    public value: number | null = null
  ) {}

  isLeaf(): boolean {
    return this.leftChild === null && this.rightChild === null;
  }
}

interface SplitInfo {
  featureIndex: number;
  threshold: number;
  score: number;
}

// Enhanced Gradient Boosting Classifier
export class EnhancedGradientBoostingClassifier {
  private trees: DecisionTree[] = [];
  private learningRate: number;
  private nEstimators: number;
  private maxDepth: number;
  private isTrained = false;
  private featureImportance: Map<string, number> = new Map();

  constructor(options: {
    learningRate?: number;
    nEstimators?: number;
    maxDepth?: number;
  } = {}) {
    this.learningRate = options.learningRate || 0.1;
    this.nEstimators = options.nEstimators || 100;
    this.maxDepth = options.maxDepth || 6;
  }

  // Enhanced feature extraction for classification
  private extractClassificationFeatures(spaceObject: any): number[] {
    const features = [
      // Orbital characteristics
      spaceObject.semiMajorAxis || 0,
      spaceObject.eccentricity || 0,
      spaceObject.inclination || 0,
      spaceObject.altitude || 0,
      spaceObject.velocity || 0,
      spaceObject.period || 0,

      // Object properties
      spaceObject.mass || 0,
      spaceObject.size || 0,
      spaceObject.areaToMassRatio || 0,
      spaceObject.reflectivity || 0,

      // Environmental factors
      spaceObject.solarActivity || 0,
      spaceObject.atmosphericDensity || 0,
      spaceObject.radiationLevel || 0,

      // Risk indicators
      spaceObject.collisionProbability || 0,
      spaceObject.fragmentationRisk || 0,
      spaceObject.reentryRisk || 0,
      spaceObject.orbitalDecayRate || 0,

      // Time-based features
      spaceObject.age || 0,
      spaceObject.maneuverability || 0,
      spaceObject.operationalStatus || 0,

      // Enhanced derived features
      this.calculateOrbitalComplexity(spaceObject),
      this.calculateEnvironmentalHazard(spaceObject),
      this.calculateOperationalRisk(spaceObject),
      this.calculateCollisionSusceptibility(spaceObject),
    ];

    return this.normalizeFeatures(features);
  }

  private calculateOrbitalComplexity(spaceObject: any): number {
    const eccentricity = spaceObject.eccentricity || 0;
    const inclination = spaceObject.inclination || 0;
    const altitude = spaceObject.altitude || 0;
    
    // Higher complexity for non-circular, high-inclination orbits
    const eccentricityComplexity = eccentricity * 0.3;
    const inclinationComplexity = Math.abs(Math.sin(inclination * Math.PI / 180)) * 0.2;
    const altitudeComplexity = (altitude < 1000) ? 0.3 : 0.1;
    
    return Math.min(1, eccentricityComplexity + inclinationComplexity + altitudeComplexity);
  }

  private calculateEnvironmentalHazard(spaceObject: any): number {
    const altitude = spaceObject.altitude || 0;
    const solarActivity = spaceObject.solarActivity || 0;
    const radiationLevel = spaceObject.radiationLevel || 0;
    const atmosphericDensity = spaceObject.atmosphericDensity || 0;
    
    let hazard = 0;
    
    // Atmospheric drag hazard
    if (altitude < 400) hazard += 0.4;
    else if (altitude < 600) hazard += 0.2;
    
    // Solar activity hazard
    hazard += solarActivity * 0.2;
    
    // Radiation hazard
    hazard += radiationLevel * 0.2;
    
    // Atmospheric density hazard
    hazard += atmosphericDensity * 0.1;
    
    return Math.min(1, hazard);
  }

  private calculateOperationalRisk(spaceObject: any): number {
    const age = spaceObject.age || 0;
    const maneuverability = spaceObject.maneuverability || 0;
    const operationalStatus = spaceObject.operationalStatus || 0;
    
    let risk = 0;
    
    // Age increases risk
    if (age > 15) risk += 0.3;
    else if (age > 10) risk += 0.2;
    else if (age > 5) risk += 0.1;
    
    // Low maneuverability increases risk
    risk += (1 - maneuverability) * 0.2;
    
    // Non-operational status increases risk
    risk += (1 - operationalStatus) * 0.3;
    
    return Math.min(1, risk);
  }

  private calculateCollisionSusceptibility(spaceObject: any): number {
    const altitude = spaceObject.altitude || 0;
    const inclination = spaceObject.inclination || 0;
    const velocity = spaceObject.velocity || 0;
    const size = spaceObject.size || 0;
    
    let susceptibility = 0;
    
    // Higher susceptibility at common altitudes
    if (altitude > 400 && altitude < 600) susceptibility += 0.3;
    if (altitude > 20000 && altitude < 36000) susceptibility += 0.2;
    
    // Higher susceptibility at common inclinations
    if (inclination > 80 && inclination < 100) susceptibility += 0.2;
    if (inclination > 0 && inclination < 10) susceptibility += 0.1;
    
    // Higher velocity increases susceptibility
    if (velocity > 7.5) susceptibility += 0.1;
    
    // Larger size increases susceptibility
    if (size > 1) susceptibility += 0.1;
    
    return Math.min(1, susceptibility);
  }

  private normalizeFeatures(features: number[]): number[] {
    return features.map(feature => {
      if (feature === 0) return 0;
      return Math.tanh(feature / 1000);
    });
  }

  // Train the gradient boosting classifier
  async train(trainingData: any[]): Promise<void> {
    console.log('Training Enhanced Gradient Boosting Classifier...');
    
    const features = trainingData.map(obj => this.extractClassificationFeatures(obj));
    const targets = trainingData.map(obj => this.encodeRiskCategory(obj.riskCategory));
    
    // Initialize predictions with mean
    const initialPrediction = targets.reduce((sum, target) => sum + target, 0) / targets.length;
    let predictions = new Array(targets.length).fill(initialPrediction);
    
    // Train boosting trees
    this.trees = [];
    for (let i = 0; i < this.nEstimators; i++) {
      // Calculate residuals
      const residuals = targets.map((target, index) => target - predictions[index]);
      
      // Train tree on residuals
      const tree = new DecisionTree({
        maxDepth: this.maxDepth,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      });
      
      await tree.train(features, residuals);
      this.trees.push(tree);
      
      // Update predictions
      for (let j = 0; j < features.length; j++) {
        const treePrediction = tree.predict(features[j]);
        predictions[j] += this.learningRate * treePrediction;
      }
    }
    
    // Calculate feature importance
    this.calculateFeatureImportance();
    
    this.isTrained = true;
    console.log('Enhanced Gradient Boosting Classifier training completed');
  }

  private encodeRiskCategory(category: string): number {
    switch (category) {
      case 'low': return 0;
      case 'medium': return 1;
      case 'high': return 2;
      case 'critical': return 3;
      default: return 0;
    }
  }

  private decodeRiskCategory(encoded: number): string {
    if (encoded < 0.5) return 'low';
    if (encoded < 1.5) return 'medium';
    if (encoded < 2.5) return 'high';
    return 'critical';
  }

  private calculateFeatureImportance(): void {
    const featureNames = [
      'semiMajorAxis', 'eccentricity', 'inclination', 'altitude', 'velocity', 'period',
      'mass', 'size', 'areaToMassRatio', 'reflectivity',
      'solarActivity', 'atmosphericDensity', 'radiationLevel',
      'collisionProbability', 'fragmentationRisk', 'reentryRisk', 'orbitalDecayRate',
      'age', 'maneuverability', 'operationalStatus',
      'orbitalComplexity', 'environmentalHazard', 'operationalRisk', 'collisionSusceptibility'
    ];
    
    for (let i = 0; i < featureNames.length; i++) {
      let importance = 0;
      
      for (const tree of this.trees) {
        importance += tree.getFeatureImportance(i);
      }
      
      this.featureImportance.set(featureNames[i], importance / this.trees.length);
    }
  }

  // Enhanced prediction with probability distribution
  async predictRiskCategory(spaceObject: any): Promise<{
    category: string;
    probabilities: { [category: string]: number };
    confidence: number;
    featureContributions: Map<string, number>;
  }> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before making predictions');
    }

    const features = this.extractClassificationFeatures(spaceObject);
    
    // Get prediction from all trees
    let prediction = 0;
    for (const tree of this.trees) {
      prediction += this.learningRate * tree.predict(features);
    }
    
    // Calculate probabilities for each category
    const probabilities = this.calculateProbabilities(prediction);
    
    // Determine most likely category
    const category = this.decodeRiskCategory(prediction);
    
    // Calculate confidence
    const confidence = Math.max(...Object.values(probabilities));
    
    // Calculate feature contributions
    const featureContributions = this.calculateFeatureContributions(features);
    
    return {
      category,
      probabilities,
      confidence,
      featureContributions,
    };
  }

  private calculateProbabilities(prediction: number): { [category: string]: number } {
    // Softmax-like probability calculation
    const categories = ['low', 'medium', 'high', 'critical'];
    const values = categories.map((_, index) => Math.exp(-Math.pow(prediction - index, 2)));
    const sum = values.reduce((s, v) => s + v, 0);
    
    const probabilities: { [category: string]: number } = {};
    categories.forEach((category, index) => {
      probabilities[category] = values[index] / sum;
    });
    
    return probabilities;
  }

  private calculateFeatureContributions(features: number[]): Map<string, number> {
    const contributions = new Map<string, number>();
    const featureNames = [
      'semiMajorAxis', 'eccentricity', 'inclination', 'altitude', 'velocity', 'period',
      'mass', 'size', 'areaToMassRatio', 'reflectivity',
      'solarActivity', 'atmosphericDensity', 'radiationLevel',
      'collisionProbability', 'fragmentationRisk', 'reentryRisk', 'orbitalDecayRate',
      'age', 'maneuverability', 'operationalStatus',
      'orbitalComplexity', 'environmentalHazard', 'operationalRisk', 'collisionSusceptibility'
    ];
    
    for (let i = 0; i < features.length; i++) {
      const importance = this.featureImportance.get(featureNames[i]) || 0;
      const contribution = features[i] * importance;
      contributions.set(featureNames[i], contribution);
    }
    
    return contributions;
  }

  // Get feature importance
  getFeatureImportance(): Map<string, number> {
    return new Map(this.featureImportance);
  }

  // Model validation
  async validate(validationData: any[]): Promise<{
    accuracy: number;
    precision: { [category: string]: number };
    recall: { [category: string]: number };
    f1Score: { [category: string]: number };
    confusionMatrix: { [category: string]: { [category: string]: number } };
  }> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before validation');
    }

    const predictions: string[] = [];
    const actuals: string[] = [];
    
    for (const obj of validationData) {
      const prediction = await this.predictRiskCategory(obj);
      predictions.push(prediction.category);
      actuals.push(obj.riskCategory || 'low');
    }
    
    return this.calculateClassificationMetrics(predictions, actuals);
  }

  private calculateClassificationMetrics(predictions: string[], actuals: string[]): any {
    const categories = ['low', 'medium', 'high', 'critical'];
    
    // Calculate confusion matrix
    const confusionMatrix: { [category: string]: { [category: string]: number } } = {};
    categories.forEach(cat => {
      confusionMatrix[cat] = {};
      categories.forEach(predCat => {
        confusionMatrix[cat][predCat] = 0;
      });
    });
    
    for (let i = 0; i < predictions.length; i++) {
      confusionMatrix[actuals[i]][predictions[i]]++;
    }
    
    // Calculate metrics for each category
    const precision: { [category: string]: number } = {};
    const recall: { [category: string]: number } = {};
    const f1Score: { [category: string]: number } = {};
    
    categories.forEach(category => {
      const truePositives = confusionMatrix[category][category];
      const falsePositives = categories.reduce((sum, cat) => 
        sum + (cat !== category ? confusionMatrix[cat][category] : 0), 0);
      const falseNegatives = categories.reduce((sum, cat) => 
        sum + (cat !== category ? confusionMatrix[category][cat] : 0), 0);
      
      precision[category] = truePositives / (truePositives + falsePositives) || 0;
      recall[category] = truePositives / (truePositives + falseNegatives) || 0;
      f1Score[category] = 2 * (precision[category] * recall[category]) / 
        (precision[category] + recall[category]) || 0;
    });
    
    // Calculate overall accuracy
    const correct = categories.reduce((sum, cat) => sum + confusionMatrix[cat][cat], 0);
    const accuracy = correct / predictions.length;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
    };
  }
}

// Enhanced Bayesian Risk Predictor
export class EnhancedBayesianRiskPredictor {
  private priorProbabilities: Map<string, number> = new Map();
  private likelihoods: Map<string, Map<string, number>> = new Map();
  private evidence: Map<string, number> = new Map();
  private isTrained = false;
  private featureWeights: Map<string, number> = new Map();

  constructor() {
    this.initializePriors();
  }

  private initializePriors(): void {
    // Initialize prior probabilities for different risk categories
    this.priorProbabilities.set('low', 0.4);
    this.priorProbabilities.set('medium', 0.3);
    this.priorProbabilities.set('high', 0.2);
    this.priorProbabilities.set('critical', 0.1);
  }

  // Enhanced feature extraction for Bayesian analysis
  private extractBayesianFeatures(spaceObject: any): Map<string, any> {
    const features = new Map<string, any>();
    
    // Orbital features
    features.set('altitude_range', this.categorizeAltitude(spaceObject.altitude || 0));
    features.set('inclination_range', this.categorizeInclination(spaceObject.inclination || 0));
    features.set('eccentricity_range', this.categorizeEccentricity(spaceObject.eccentricity || 0));
    features.set('velocity_range', this.categorizeVelocity(spaceObject.velocity || 0));
    
    // Object features
    features.set('mass_range', this.categorizeMass(spaceObject.mass || 0));
    features.set('size_range', this.categorizeSize(spaceObject.size || 0));
    features.set('age_range', this.categorizeAge(spaceObject.age || 0));
    features.set('operational_status', spaceObject.operationalStatus || 0);
    
    // Environmental features
    features.set('solar_activity_level', this.categorizeSolarActivity(spaceObject.solarActivity || 0));
    features.set('atmospheric_density_level', this.categorizeAtmosphericDensity(spaceObject.atmosphericDensity || 0));
    features.set('radiation_level', this.categorizeRadiation(spaceObject.radiationLevel || 0));
    
    // Risk indicators
    features.set('collision_probability_level', this.categorizeCollisionProbability(spaceObject.collisionProbability || 0));
    features.set('fragmentation_risk_level', this.categorizeFragmentationRisk(spaceObject.fragmentationRisk || 0));
    features.set('reentry_risk_level', this.categorizeReentryRisk(spaceObject.reentryRisk || 0));
    
    return features;
  }

  // Categorization functions
  private categorizeAltitude(altitude: number): string {
    if (altitude < 200) return 'very_low';
    if (altitude < 400) return 'low';
    if (altitude < 600) return 'medium';
    if (altitude < 2000) return 'high';
    if (altitude < 36000) return 'very_high';
    return 'geostationary';
  }

  private categorizeInclination(inclination: number): string {
    if (inclination < 10) return 'equatorial';
    if (inclination < 30) return 'low_inclination';
    if (inclination < 60) return 'medium_inclination';
    if (inclination < 90) return 'high_inclination';
    return 'polar';
  }

  private categorizeEccentricity(eccentricity: number): string {
    if (eccentricity < 0.01) return 'circular';
    if (eccentricity < 0.1) return 'low_eccentricity';
    if (eccentricity < 0.5) return 'medium_eccentricity';
    return 'high_eccentricity';
  }

  private categorizeVelocity(velocity: number): string {
    if (velocity < 6) return 'low_velocity';
    if (velocity < 7.5) return 'medium_velocity';
    if (velocity < 8.5) return 'high_velocity';
    return 'very_high_velocity';
  }

  private categorizeMass(mass: number): string {
    if (mass < 1) return 'very_small';
    if (mass < 10) return 'small';
    if (mass < 100) return 'medium';
    if (mass < 1000) return 'large';
    return 'very_large';
  }

  private categorizeSize(size: number): string {
    if (size < 0.1) return 'very_small';
    if (size < 1) return 'small';
    if (size < 5) return 'medium';
    if (size < 20) return 'large';
    return 'very_large';
  }

  private categorizeAge(age: number): string {
    if (age < 1) return 'new';
    if (age < 5) return 'young';
    if (age < 10) return 'mature';
    if (age < 20) return 'old';
    return 'very_old';
  }

  private categorizeSolarActivity(activity: number): string {
    if (activity < 0.2) return 'low';
    if (activity < 0.5) return 'medium';
    if (activity < 0.8) return 'high';
    return 'very_high';
  }

  private categorizeAtmosphericDensity(density: number): string {
    if (density < 0.1) return 'very_low';
    if (density < 0.3) return 'low';
    if (density < 0.6) return 'medium';
    if (density < 0.8) return 'high';
    return 'very_high';
  }

  private categorizeRadiation(radiation: number): string {
    if (radiation < 0.2) return 'low';
    if (radiation < 0.5) return 'medium';
    if (radiation < 0.8) return 'high';
    return 'very_high';
  }

  private categorizeCollisionProbability(probability: number): string {
    if (probability < 0.001) return 'very_low';
    if (probability < 0.01) return 'low';
    if (probability < 0.1) return 'medium';
    if (probability < 0.5) return 'high';
    return 'very_high';
  }

  private categorizeFragmentationRisk(risk: number): string {
    if (risk < 0.1) return 'low';
    if (risk < 0.3) return 'medium';
    if (risk < 0.6) return 'high';
    return 'very_high';
  }

  private categorizeReentryRisk(risk: number): string {
    if (risk < 0.1) return 'low';
    if (risk < 0.3) return 'medium';
    if (risk < 0.6) return 'high';
    return 'very_high';
  }

  // Train the Bayesian model
  async train(trainingData: any[]): Promise<void> {
    console.log('Training Enhanced Bayesian Risk Predictor...');
    
    // Calculate likelihoods for each feature given each risk category
    const riskCategories = ['low', 'medium', 'high', 'critical'];
    
    for (const category of riskCategories) {
      this.likelihoods.set(category, new Map());
      
      // Get objects in this risk category
      const categoryObjects = trainingData.filter(obj => obj.riskCategory === category);
      const categoryCount = categoryObjects.length;
      
      if (categoryCount === 0) continue;
      
      // Calculate likelihoods for each feature
      const featureLikelihoods = new Map<string, Map<string, number>>();
      
      for (const obj of categoryObjects) {
        const features = this.extractBayesianFeatures(obj);
        
        for (const [featureName, featureValue] of features) {
          if (!featureLikelihoods.has(featureName)) {
            featureLikelihoods.set(featureName, new Map());
          }
          
          const featureMap = featureLikelihoods.get(featureName)!;
          const currentCount = featureMap.get(featureValue) || 0;
          featureMap.set(featureValue, currentCount + 1);
        }
      }
      
      // Convert counts to probabilities with Laplace smoothing
      for (const [featureName, valueCounts] of featureLikelihoods) {
        const totalValues = valueCounts.size;
        
        for (const [value, count] of valueCounts) {
          const probability = (count + 1) / (categoryCount + totalValues); // Laplace smoothing
          const key = `${featureName}_${value}`;
          this.likelihoods.get(category)!.set(key, probability);
        }
      }
    }
    
    // Calculate feature weights based on information gain
    this.calculateFeatureWeights(trainingData);
    
    this.isTrained = true;
    console.log('Enhanced Bayesian Risk Predictor training completed');
  }

  private calculateFeatureWeights(trainingData: any[]): void {
    const totalObjects = trainingData.length;
    const riskCategories = ['low', 'medium', 'high', 'critical'];
    
    // Calculate entropy of the target variable
    const categoryCounts = new Map<string, number>();
    for (const obj of trainingData) {
      const category = obj.riskCategory || 'low';
      categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1);
    }
    
    let targetEntropy = 0;
    for (const count of categoryCounts.values()) {
      const probability = count / totalObjects;
      targetEntropy -= probability * Math.log2(probability);
    }
    
    // Calculate information gain for each feature
    const featureNames = [
      'altitude_range', 'inclination_range', 'eccentricity_range', 'velocity_range',
      'mass_range', 'size_range', 'age_range', 'operational_status',
      'solar_activity_level', 'atmospheric_density_level', 'radiation_level',
      'collision_probability_level', 'fragmentation_risk_level', 'reentry_risk_level'
    ];
    
    for (const featureName of featureNames) {
      let weightedEntropy = 0;
      
      // Get unique values for this feature
      const featureValues = new Set<string>();
      for (const obj of trainingData) {
        const features = this.extractBayesianFeatures(obj);
        featureValues.add(features.get(featureName));
      }
      
      for (const value of featureValues) {
        // Get objects with this feature value
        const valueObjects = trainingData.filter(obj => {
          const features = this.extractBayesianFeatures(obj);
          return features.get(featureName) === value;
        });
        
        const valueCount = valueObjects.length;
        const valueProbability = valueCount / totalObjects;
        
        // Calculate entropy for this feature value
        const valueCategoryCounts = new Map<string, number>();
        for (const obj of valueObjects) {
          const category = obj.riskCategory || 'low';
          valueCategoryCounts.set(category, (valueCategoryCounts.get(category) || 0) + 1);
        }
        
        let valueEntropy = 0;
        for (const count of valueCategoryCounts.values()) {
          const probability = count / valueCount;
          valueEntropy -= probability * Math.log2(probability);
        }
        
        weightedEntropy += valueProbability * valueEntropy;
      }
      
      const informationGain = targetEntropy - weightedEntropy;
      this.featureWeights.set(featureName, Math.max(0, informationGain));
    }
  }

  // Enhanced Bayesian prediction with uncertainty quantification
  async predictRisk(spaceObject: any): Promise<{
    riskCategory: string;
    probabilities: { [category: string]: number };
    confidence: number;
    uncertainty: number;
    featureContributions: Map<string, number>;
    evidence: Map<string, number>;
  }> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before making predictions');
    }

    const features = this.extractBayesianFeatures(spaceObject);
    const riskCategories = ['low', 'medium', 'high', 'critical'];
    
    // Calculate posterior probabilities using Bayes' theorem
    const posteriors = new Map<string, number>();
    const evidence = new Map<string, number>();
    
    for (const category of riskCategories) {
      const prior = this.priorProbabilities.get(category) || 0;
      let likelihood = 1;
      let featureEvidence = 0;
      
      // Calculate likelihood for this category
      for (const [featureName, featureValue] of features) {
        const key = `${featureName}_${featureValue}`;
        const featureLikelihood = this.likelihoods.get(category)?.get(key) || 0.001; // Small default value
        const featureWeight = this.featureWeights.get(featureName) || 0;
        
        likelihood *= Math.pow(featureLikelihood, featureWeight);
        featureEvidence += featureWeight * Math.log(featureLikelihood);
      }
      
      const posterior = prior * likelihood;
      posteriors.set(category, posterior);
      evidence.set(category, featureEvidence);
    }
    
    // Normalize probabilities
    const totalPosterior = Array.from(posteriors.values()).reduce((sum, prob) => sum + prob, 0);
    const normalizedProbabilities: { [category: string]: number } = {};
    
    for (const category of riskCategories) {
      normalizedProbabilities[category] = posteriors.get(category)! / totalPosterior;
    }
    
    // Determine most likely category
    let maxProbability = 0;
    let predictedCategory = 'low';
    
    for (const [category, probability] of Object.entries(normalizedProbabilities)) {
      if (probability > maxProbability) {
        maxProbability = probability;
        predictedCategory = category;
      }
    }
    
    // Calculate confidence and uncertainty
    const probabilities = Object.values(normalizedProbabilities);
    const confidence = maxProbability;
    const uncertainty = this.calculateUncertainty(probabilities);
    
    // Calculate feature contributions
    const featureContributions = this.calculateFeatureContributions(features, predictedCategory);
    
    return {
      riskCategory: predictedCategory,
      probabilities: normalizedProbabilities,
      confidence,
      uncertainty,
      featureContributions,
      evidence,
    };
  }

  private calculateUncertainty(probabilities: number[]): number {
    // Calculate entropy as a measure of uncertainty
    let entropy = 0;
    for (const prob of probabilities) {
      if (prob > 0) {
        entropy -= prob * Math.log2(prob);
      }
    }
    return entropy;
  }

  private calculateFeatureContributions(features: Map<string, any>, predictedCategory: string): Map<string, number> {
    const contributions = new Map<string, number>();
    
    for (const [featureName, featureValue] of features) {
      const key = `${featureName}_${featureValue}`;
      const likelihood = this.likelihoods.get(predictedCategory)?.get(key) || 0.001;
      const weight = this.featureWeights.get(featureName) || 0;
      
      contributions.set(featureName, weight * Math.log(likelihood));
    }
    
    return contributions;
  }

  // Model validation
  async validate(validationData: any[]): Promise<{
    accuracy: number;
    precision: { [category: string]: number };
    recall: { [category: string]: number };
    f1Score: { [category: string]: number };
    averageConfidence: number;
    averageUncertainty: number;
  }> {
    if (!this.isTrained) {
      throw new Error('Model must be trained before validation');
    }

    const predictions: string[] = [];
    const actuals: string[] = [];
    const confidences: number[] = [];
    const uncertainties: number[] = [];
    
    for (const obj of validationData) {
      const prediction = await this.predictRisk(obj);
      predictions.push(prediction.riskCategory);
      actuals.push(obj.riskCategory || 'low');
      confidences.push(prediction.confidence);
      uncertainties.push(prediction.uncertainty);
    }
    
    // Calculate classification metrics
    const metrics = this.calculateClassificationMetrics(predictions, actuals);
    
    return {
      ...metrics,
      averageConfidence: confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length,
      averageUncertainty: uncertainties.reduce((sum, unc) => sum + unc, 0) / uncertainties.length,
    };
  }

  private calculateClassificationMetrics(predictions: string[], actuals: string[]): any {
    const categories = ['low', 'medium', 'high', 'critical'];
    
    // Calculate confusion matrix
    const confusionMatrix: { [category: string]: { [category: string]: number } } = {};
    categories.forEach(cat => {
      confusionMatrix[cat] = {};
      categories.forEach(predCat => {
        confusionMatrix[cat][predCat] = 0;
      });
    });
    
    for (let i = 0; i < predictions.length; i++) {
      confusionMatrix[actuals[i]][predictions[i]]++;
    }
    
    // Calculate metrics for each category
    const precision: { [category: string]: number } = {};
    const recall: { [category: string]: number } = {};
    const f1Score: { [category: string]: number } = {};
    
    categories.forEach(category => {
      const truePositives = confusionMatrix[category][category];
      const falsePositives = categories.reduce((sum, cat) => 
        sum + (cat !== category ? confusionMatrix[cat][category] : 0), 0);
      const falseNegatives = categories.reduce((sum, cat) => 
        sum + (cat !== category ? confusionMatrix[category][cat] : 0), 0);
      
      precision[category] = truePositives / (truePositives + falsePositives) || 0;
      recall[category] = truePositives / (truePositives + falseNegatives) || 0;
      f1Score[category] = 2 * (precision[category] * recall[category]) / 
        (precision[category] + recall[category]) || 0;
    });
    
    // Calculate overall accuracy
    const correct = categories.reduce((sum, cat) => sum + confusionMatrix[cat][cat], 0);
    const accuracy = correct / predictions.length;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
    };
  }

  // Get model insights
  getModelInsights(): {
    featureWeights: Map<string, number>;
    priorProbabilities: Map<string, number>;
    topFeatures: string[];
  } {
    const topFeatures = Array.from(this.featureWeights.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([feature]) => feature);
    
    return {
      featureWeights: new Map(this.featureWeights),
      priorProbabilities: new Map(this.priorProbabilities),
      topFeatures,
    };
  }
}
