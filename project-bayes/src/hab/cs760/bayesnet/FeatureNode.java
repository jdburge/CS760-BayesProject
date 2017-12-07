package hab.cs760.bayesnet;

import com.sun.istack.internal.Nullable;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by hannah on 10/30/17.
 */
public class FeatureNode extends Node {
	public Edge labelNodeEdge;
	private Map<String, Map<String, Map<String, Double>>> probabilities;

	FeatureNode(NominalFeature feature) {
		super(feature);
		this.labelNodeEdge = null;
		probabilities = new HashMap<>();
	}

	@Override
	protected void calculateProbabilitiesForThisNode(List<Instance> instances) {
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;

		Node treeParent = getTreeParent();
		for (String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			probabilities.put(labelFeatureValue, block);

			if (treeParent != null) {
				NominalFeature treeParentFeature = treeParent.feature;

				for (String treeParentFeatureValue : treeParentFeature.possibleValues) {
					Map<String, Double> row = rowWithCriteria(instances, new InstanceCounter
							.Criterion(labelFeature, labelFeatureValue), new InstanceCounter
							.Criterion(treeParentFeature, treeParentFeatureValue));
					block.put(treeParentFeatureValue, row);
				}
			} else {
				Map<String, Double> row = rowWithCriteria(instances, new InstanceCounter.Criterion
						(labelFeature, labelFeatureValue));
				block.put("", row);
			}
		}
	}

	private Map<String, Double> rowWithCriteria(List<Instance> instances, InstanceCounter
			.Criterion... criteria) {
		Map<String, Double> row = new HashMap<>();

		for (String featureValue : feature.possibleValues) {
			double probability = InstanceCounter.probabilityOfCriterionGivenCriteria(feature,
					featureValue, instances, criteria);
			row.put(featureValue, probability);
		}
		return row;
	}

	@Nullable
	private Node getTreeParent() {
		for (Edge edge : connectedEdges) {
			if (edge.end() == this) {
				return edge.start();
			}
		}
		return null;
	}

	@Override
	public double probabilityOf(String labelNodeValue, String treeParentValue, String
			thisFeatureValue) {
		if (getTreeParent() != null) {
			assert getTreeParent().feature.possibleValues.contains(treeParentValue);
			return probabilities.get(labelNodeValue).get(treeParentValue).get(thisFeatureValue);
		}
		return probabilities.get(labelNodeValue).get("").get(thisFeatureValue);
	}

	public void makeFair1() {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap();
		FeatureNode sensitiveNode = (FeatureNode)getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb = -1.0;
		double probSensitiveFeature = -1.0;
		double newProb = -1.0;
		double weightedAverage = -1.0;

		for(String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);

			for(String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				Map<String, Double> row = new HashMap();
				probSensitiveFeature = getProbOfSensitiveFeature(sensitiveNode, sensitiveFeatureValue);

				for(String thisFeatureValue : this.feature.possibleValues) {
					baseProb = probabilityOf(labelFeatureValue,sensitiveFeatureValue,thisFeatureValue);
					newProb = (baseProb*probSensitiveFeature) + ((1-probSensitiveFeature)*(1.0/this.feature.possibleValues.size()));
					row.put(thisFeatureValue, newProb);
				}
				block.put(sensitiveFeatureValue, row);
			}	
		}

		probabilities = newProbabilities; 
	}

	public void makeFair2() {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap();
		FeatureNode sensitiveNode = (FeatureNode)getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb = -1.0;
		double probSensitiveFeature = -1.0;
		double newProb = 0.0;
		double weightedAverage = -1.0;
		Map<String, Double> row = null;

		for(String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);
			for(String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				row = new HashMap();
				block.put(sensitiveFeatureValue, row);
			}

			for(String thisFeatureValue : this.feature.possibleValues) {
				newProb = 0.0;
				for(String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
					probSensitiveFeature = getProbOfSensitiveFeature(sensitiveNode, sensitiveFeatureValue);
					baseProb = probabilityOf(labelFeatureValue,sensitiveFeatureValue,thisFeatureValue);
					newProb += baseProb;
				}
				newProb = newProb/((double)sensitiveFeature.possibleValues.size());

				for(String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
					row = block.get(sensitiveFeatureValue);
					baseProb = probabilityOf(labelFeatureValue,sensitiveFeatureValue,thisFeatureValue);
					row.put(thisFeatureValue, newProb);
				}
			}

		}

		probabilities = newProbabilities; 

	}

	public void makeFair3(List<Instance> instances) {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap();
		FeatureNode sensitiveNode = (FeatureNode)getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb = -1.0;
		double probSensitiveFeature = -1.0;
		double newProb = -1.0;
		double weightedAverage = -1.0;

		for(String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);

			for(String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				Map<String, Double> row = new HashMap();
				probSensitiveFeature = getProbOfSensitiveFeature(sensitiveNode, sensitiveFeatureValue);
				weightedAverage = getWeightedAverage(labelFeatureValue,sensitiveFeatureValue,sensitiveNode, instances);

				for(String thisFeatureValue : this.feature.possibleValues) {
					baseProb = probabilityOf(labelFeatureValue,sensitiveFeatureValue,thisFeatureValue);
					newProb = (baseProb*probSensitiveFeature) + ((1-probSensitiveFeature)*(weightedAverage));
					row.put(thisFeatureValue, newProb);
				}

				block.put(sensitiveFeatureValue, row);
			}

		}

		probabilities = newProbabilities;
		normalizeProbTable();
	}

	private double getProbOfSensitiveFeature(FeatureNode sensitiveNode, String choice) {
		double runningProb =0.0;
		NominalFeature labelFeature = labelNodeEdge.start().feature;
		for(String labelFeatureValue : labelFeature.possibleValues) {
			runningProb += sensitiveNode.probabilityOf(labelFeatureValue,"",choice);
		}
		return runningProb/labelFeature.possibleValues.size();
	}

	/**
	 * Returns the probability of a choice for this feature, knowing that you don't
	 * have a particular sensitive attribute
	 * @param sensitiveNode
	 * @param sensitiveFeatureValue
	 * @param thisFeatureValue
	 * @return
	 */
	private double secondOrderWeights(FeatureNode sensitiveNode, String sensitiveFeatureValue, String thisFeatureValue, List<Instance> instances) {
		double runningProb = 0.0;
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Criterion crit = null;

		for(String curSensitiveFeatureValue : sensitiveFeature.possibleValues) {
			if(!sensitiveFeatureValue.equals(curSensitiveFeatureValue)) {
				crit = new Criterion(sensitiveFeature, curSensitiveFeatureValue);	
				runningProb += InstanceCounter.probabilityOfCriterionGivenCriteria(this.feature, thisFeatureValue, instances, crit);
			}
		}

		return runningProb;
	}

	private double getWeightedAverage(String labelFeatureValue, String sensitiveFeatureValue, FeatureNode sensitiveFeatureNode, List<Instance> instances) {
		NominalFeature sensitiveFeature = sensitiveFeatureNode.feature;
		double runningProb = 0.0;
		for(String otherSensFeatureVal : sensitiveFeature.possibleValues) {
			if(!otherSensFeatureVal.equals(sensitiveFeatureValue)) {
				for(String thisFeatureValue : this.feature.possibleValues) {
					runningProb += (probabilityOf(labelFeatureValue,otherSensFeatureVal, thisFeatureValue) *
							secondOrderWeights(sensitiveFeatureNode,sensitiveFeatureValue,thisFeatureValue, instances));
				}
			}
		}
		return runningProb/(sensitiveFeature.possibleValues.size()-1);
	}

	private void normalizeProbTable() {
		FeatureNode sensitiveNode = (FeatureNode)getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double runningTotal = 0.0;
		double average = 0.0;
		double normalizedValue=0.0;
		Map<String, Map<String, Double>> block = null;
		Map<String, Double> row = null;

		for(String labelFeatureValue: labelFeature.possibleValues) {
			block = probabilities.get(labelFeatureValue);
			for(String sensitiveFeatureValue: sensitiveFeature.possibleValues) {
				row = block.get(sensitiveFeatureValue);
				runningTotal=0.0;
				for(String thisFeatureValue: this.feature.possibleValues) {
					runningTotal += probabilityOf(labelFeatureValue,sensitiveFeatureValue,thisFeatureValue);
				}

				average = runningTotal/((double)this.feature.possibleValues.size());

				for(String thisFeatureValue: this.feature.possibleValues) {
					normalizedValue = probabilityOf(labelFeatureValue,sensitiveFeatureValue,thisFeatureValue)/average;
					row.put(thisFeatureValue, normalizedValue);
				}
			}


		}
	}
}
