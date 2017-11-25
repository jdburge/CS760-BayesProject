package hab.cs760.machinelearning;

/**
 * Created by hannah on 9/29/17.
 */
public abstract class Feature {
	public final String name;
	public final int index;

	Feature(String name, int index) {
		this.name = name;
		this.index = index;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		Feature feature = (Feature) o;

		if (index != feature.index) return false;
		return name != null ? name.equals(feature.name) : feature.name == null;
	}

	@Override
	public int hashCode() {
		int result = name != null ? name.hashCode() : 0;
		result = 31 * result + index;
		return result;
	}

	@Override
	public String toString() {
		return String.format("Name: %s, Index: %s", name, index);
	}
}
