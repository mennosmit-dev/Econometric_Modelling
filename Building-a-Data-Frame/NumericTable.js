import java.util.*;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Represents a numeric table with support for column and row-based operations.
 */
public class NumericTable implements TableData<Double> {

    private final Map<Integer, LinkedHashMap<String, Double>> data;
    private final double[][] initialData;

    /**
     * Constructs a new NumericTable with headers and data.
     *
     * @param headers the column headers
     * @param values  the table data
     */
    public NumericTable(List<String> headers, double[][] values) {
        this.data = new LinkedHashMap<>();
        for (int i = 0; i < values.length; i++) {
            LinkedHashMap<String, Double> rowMap = new LinkedHashMap<>();
            for (int j = 0; j < headers.size(); j++) {
                rowMap.put(headers.get(j), values[i][j]);
            }
            this.data.put(i, rowMap);
        }
        this.initialData = values;
    }

    @Override
    public void updateValue(int row, String column, Double value) {
        validateRowIndex(row);
        if (!data.get(row).containsKey(column)) {
            throw new IllegalArgumentException("Column '" + column + "' not found in row " + row);
        }
        data.get(row).put(column, value);
    }

    @Override
    public Double retrieveValue(int row, String column) {
        validateRowIndex(row);
        if (!data.get(row).containsKey(column)) {
            throw new IllegalArgumentException("Column '" + column + "' not found in row " + row);
        }
        return data.get(row).get(column);
    }

    @Override
    public int rowCount() {
        return data.size();
    }

    @Override
    public int columnCount() {
        return data.get(0).size();
    }

    @Override
    public List<String> getHeaders() {
        return new ArrayList<>(data.get(0).keySet());
    }

    @Override
    public VectorData<Double> getRowVector(int row) {
        validateRowIndex(row);
        return new NumericVector(data.get(row), "row_" + row);
    }

    @Override
    public VectorData<Double> getColumnVector(String column) {
        if (!getHeaders().contains(column)) {
            throw new IllegalArgumentException("Column '" + column + "' not found.");
        }
        LinkedHashMap<String, Double> columnData = new LinkedHashMap<>();
        for (int i = 0; i < data.size(); i++) {
            columnData.put("row_" + i, data.get(i).get(column));
        }
        return new NumericVector(columnData, column);
    }

    @Override
    public List<VectorData<Double>> getAllRows() {
        List<VectorData<Double>> rows = new ArrayList<>();
        for (Integer rowIndex : data.keySet()) {
            rows.add(getRowVector(rowIndex));
        }
        return rows;
    }

    @Override
    public List<VectorData<Double>> getAllColumns() {
        List<VectorData<Double>> columns = new ArrayList<>();
        for (String header : getHeaders()) {
            columns.add(getColumnVector(header));
        }
        return columns;
    }

    @Override
    public TableData<Double> extend(int newRowCount, List<String> newColumns) {
        if (newRowCount < 0) {
            throw new IllegalArgumentException("Row count must be non-negative.");
        }

        List<String> updatedHeaders = new ArrayList<>(getHeaders());
        updatedHeaders.addAll(newColumns);

        double[][] expandedData = new double[initialData.length + newRowCount][updatedHeaders.size()];

        for (int i = 0; i < initialData.length; i++) {
            for (int j = 0; j < getHeaders().size(); j++) {
                expandedData[i][j] = data.get(i).get(getHeaders().get(j));
            }
        }

        for (int i = initialData.length; i < expandedData.length; i++) {
            for (int j = getHeaders().size(); j < updatedHeaders.size(); j++) {
                expandedData[i][j] = 0.0;
            }
        }

        return new NumericTable(updatedHeaders, expandedData);
    }

    public List<String> reorderColumns(Collection<String> columnsToKeep) {
        List<String> reordered = new ArrayList<>();
        for (String column : getHeaders()) {
            if (columnsToKeep.contains(column)) {
                reordered.add(column);
            }
        }
        return reordered;
    }

    @Override
    public TableData<Double> filterColumns(Collection<String> columnsToKeep) {
        if (!getHeaders().containsAll(columnsToKeep)) {
            throw new IllegalArgumentException("Some specified columns do not exist in the table.");
        }

        List<String> filteredHeaders = reorderColumns(columnsToKeep);
        double[][] filteredData = new double[initialData.length][filteredHeaders.size()];

        for (int i = 0; i < initialData.length; i++) {
            for (int j = 0; j < filteredHeaders.size(); j++) {
                filteredData[i][j] = data.get(i).get(filteredHeaders.get(j));
            }
        }

        return new NumericTable(filteredHeaders, filteredData);
    }

    @Override
    public TableData<Double> filterRows(Predicate<VectorData<Double>> predicate) {
        List<List<Double>> keptRows = new ArrayList<>();

        for (int i = 0; i < initialData.length; i++) {
            VectorData<Double> row = getRowVector(i);
            if (predicate.test(row)) {
                keptRows.add(new ArrayList<>(row.getValues()));
            }
        }

        double[][] result = new double[keptRows.size()][columnCount()];
        for (int i = 0; i < keptRows.size(); i++) {
            for (int j = 0; j < keptRows.get(i).size(); j++) {
                result[i][j] = keptRows.get(i).get(j);
            }
        }

        return new NumericTable(getHeaders(), result);
    }

    @Override
    public TableData<Double> addComputedColumn(String columnName, Function<VectorData<Double>, Double> calculation) {
        NumericTable extended = (NumericTable) this.extend(0, Collections.singletonList(columnName));

        for (int i = 0; i < initialData.length; i++) {
            Double value = calculation.apply(getRowVector(i));
            extended.updateValue(i, columnName, value);
        }

        return extended;
    }

    @Override
    public VectorData<Double> summarizeColumn(String name, BinaryOperator<Double> aggregator) {
        LinkedHashMap<String, Double> summary = new LinkedHashMap<>();

        for (String header : getHeaders()) {
            double result = 0.0;
            boolean first = true;
            for (int i = 0; i < initialData.length; i++) {
                double value = getColumnVector(header).getValue("row_" + i);
                result = first ? value : aggregator.apply(result, value);
                first = false;
            }
            summary.put(header, result);
        }

        return new NumericVector(summary, name);
    }

    @Override
    public TableAnalytics statistics() {
        return new DataAnalytics(this);
    }

    private void validateRowIndex(int row) {
        if (row < 0 || row >= rowCount()) {
            throw new IndexOutOfBoundsException("Row index " + row + " is out of bounds.");
        }
    }
}
