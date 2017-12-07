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

	void makeFair() {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap<>();
		FeatureNode sensitiveNode = (FeatureNode) getTreeParent();
		if (sensitiveNode == null) return;

		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = labelNodeEdge.start();

		double baseProb;
		double probSensitiveFeature;
		double newProb;
		double weightedAverage;

		for (String labelFeatureValue : labelNode.feature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);

			for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				Map<String, Double> row = new HashMap<>();
				probSensitiveFeature = getProbOfSensitiveFeature(sensitiveNode,
						sensitiveFeatureValue);
				weightedAverage = getWeightedAverage(labelFeatureValue, sensitiveFeatureValue,
						sensitiveFeature);

				for (String thisFeatureValue : this.feature.possibleValues) {
					baseProb = probabilityOf(labelFeatureValue, sensitiveFeatureValue,
							thisFeatureValue);
					newProb = (baseProb * probSensitiveFeature) + ((1 - probSensitiveFeature) *
							weightedAverage);
					row.put(thisFeatureValue, newProb);
				}

				block.put(sensitiveFeatureValue, row);
			}
		}
		probabilities = newProbabilities;
	}

	private double getProbOfSensitiveFeature(FeatureNode sensitiveNode, String choice) {
		double runningProb = 0.0;
		NominalFeature labelFeature = labelNodeEdge.start().feature;
		for (String labelFeatureValue : labelFeature.possibleValues) {
			runningProb += sensitiveNode.probabilityOf(labelFeatureValue, "", choice);
		}
		return runningProb;
	}

	private double getWeightedAverage(String labelFeatureValue, String sensitiveFeatureValue,
									  NominalFeature sensitiveFeature) {
		double runningProb = 0.0;
		for (String otherSensFeatureVal : sensitiveFeature.possibleValues) {
			if (!otherSensFeatureVal.equals(sensitiveFeatureValue)) {
				for (String thisFeatureValue : this.feature.possibleValues) {
					runningProb += probabilityOf(labelFeatureValue, sensitiveFeatureValue,
							thisFeatureValue);
				}
			}
		}
		int sensitiveFeatureValueCount = sensitiveFeature.possibleValues.size();
		int featureValueCount = feature.possibleValues.size();
		return runningProb / (sensitiveFeatureValueCount * featureValueCount);
	}
}
