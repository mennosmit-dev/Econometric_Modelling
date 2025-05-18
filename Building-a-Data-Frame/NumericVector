import java.util.*;

/**
 * A simple implementation of VectorData that holds numeric values with named entries.
 */
public class NumericVector implements VectorData<Double> {

    private final Map<String, Double> elements;
    private final String vectorName;

    /**
     * Constructs a new NumericVector with specified elements and name.
     *
     * @param elements   the key-value pairs representing vector entries
     * @param vectorName the name of the vector
     */
    public NumericVector(LinkedHashMap<String, Double> elements, String vectorName) {
        this.elements = new LinkedHashMap<>(elements);  // Preserve insertion order
        this.vectorName = vectorName;
    }

    @Override
    public String getName() {
        return this.vectorName;
    }

    @Override
    public List<String> getEntryNames() {
        return new ArrayList<>(elements.keySet());
    }

    @Override
    public Double getValue(String entry) {
        return elements.get(entry);
    }

    @Override
    public List<Double> getValues() {
        return new ArrayList<>(elements.values());
    }

    @Override
    public Map<String, Double> asMap() {
        return Collections.unmodifiableMap(elements);  // Prevent external modification
    }
}
