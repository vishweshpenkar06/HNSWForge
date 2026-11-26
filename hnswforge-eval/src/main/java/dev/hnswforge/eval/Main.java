package dev.hnswforge.eval;

import dev.hnswforge.index.Metric;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Command-line entry point for dataset evaluation.
 *
 * <p>Supported subcommands are {@code groundtruth}, {@code recall} and
 * {@code sweep}. Bad arguments print usage to standard error and produce exit code {@code 2};
 * I/O and data failures produce exit code {@code 1}. Successful commands print
 * one line to standard output and return {@code 0}. The dispatch is a switch so
 * further subcommands can be added without restructuring the parser.
 *
 * <p>Instances are never created. The methods use the shared standard streams
 * and are therefore not safe for concurrent invocations.
 */
public final class Main {

  private Main() {
  }

  /**
   * Runs the CLI and terminates the JVM with the returned exit code.
   *
   * <p>Not thread-safe for concurrent invocations because it uses the shared standard streams.
   *
   * @param args the subcommand and its options; must not be {@code null}
   */
  public static void main(String[] args) {
    System.exit(run(args));
  }

  /**
   * Runs the CLI without terminating the JVM, for use by tests and embedders.
   *
   * <p>Not thread-safe for concurrent invocations because it uses the shared standard streams.
   *
   * @param args the subcommand and its options; may be {@code null} or empty to request usage
   * @return {@code 0} on success, {@code 2} on bad arguments, {@code 1} on I/O or data failure
   */
  public static int run(String[] args) {
    if (args == null || args.length == 0) {
      return usageError("Missing subcommand");
    }
    return switch (args[0]) {
      case "groundtruth" -> runGroundTruth(args);
      case "recall" -> runRecall(args);
      case "sweep" -> runSweep(args);
      default -> usageError("Unknown subcommand: " + args[0]);
    };
  }

  /**
   * Computes and writes exact ground truth.
   *
   * @param args the full argument array, including the subcommand at index zero
   * @return the exit code
   */
  private static int runGroundTruth(String[] args) {
    try {
      Map<String, String> options = parseOptions(args, Set.of("--base", "--queries", "--metric", "--k", "--out"));
      Path base = Path.of(required(options, "--base"));
      Path queries = Path.of(required(options, "--queries"));
      Metric metric = parseMetric(required(options, "--metric"));
      int k = positiveInt(required(options, "--k"), "--k");
      Path out = Path.of(required(options, "--out"));
      int[][] truth = GroundTruth.compute(FvecsReader.read(base), FvecsReader.read(queries), metric, k);
      GroundTruth.writeIvecs(out, truth);
      System.out.println("Wrote " + truth.length + " ground-truth rows to " + out);
      return 0;
    } catch (IllegalArgumentException e) {
      return usageError(e.getMessage());
    } catch (IOException e) {
      return failure(e.getMessage());
    }
  }

  /**
   * Computes recall of a saved result-id text file against saved ground truth.
   *
   * @param args the full argument array, including the subcommand at index zero
   * @return the exit code
   */
  private static int runRecall(String[] args) {
    try {
      Map<String, String> options = parseOptions(args, Set.of("--truth", "--results", "--k"));
      Path truthPath = Path.of(required(options, "--truth"));
      Path resultsPath = Path.of(required(options, "--results"));
      int k = positiveInt(required(options, "--k"), "--k");
      int[][] truth = GroundTruth.readIvecs(truthPath);
      var results = Recall.readResultIds(resultsPath);
      double recall = Recall.recallAtK(truth, results, k);
      System.out.printf(Locale.ROOT, "Recall@%d: %.6f%n", k, recall);
      return 0;
    } catch (IllegalArgumentException e) {
      return usageError(e.getMessage());
    } catch (IOException e) {
      return failure(e.getMessage());
    }
  }

  /**
   * Runs a parameter sweep and writes its CSV report.
   *
   * @param args the full argument array, including the subcommand at index zero
   * @return the exit code
   */
  private static int runSweep(String[] args) {
    try {
      Sweep.run(args);
      return 0;
    } catch (IllegalArgumentException e) {
      return usageError(e.getMessage());
    } catch (IllegalStateException | IOException e) {
      return failure(e.getMessage());
    }
  }

  /**
   * Parses {@code --flag value} pairs after the subcommand.
   *
   * @param args the full argument array
   * @param allowed the flags accepted by the current subcommand
   * @return the parsed flag-to-value map
   * @throws IllegalArgumentException on unknown, duplicate, or valueless flags
   */
  private static Map<String, String> parseOptions(String[] args, Set<String> allowed) {
    Map<String, String> options = new HashMap<>();
    for (int i = 1; i < args.length; i++) {
      String flag = args[i];
      if (!flag.startsWith("--") || !allowed.contains(flag)) {
        throw new IllegalArgumentException("Unknown option: " + flag);
      }
      if (options.containsKey(flag)) {
        throw new IllegalArgumentException("Duplicate option: " + flag);
      }
      if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
        throw new IllegalArgumentException("Missing value for " + flag);
      }
      options.put(flag, args[++i]);
    }
    return options;
  }

  /**
   * Looks up a required option.
   *
   * @param options the parsed options
   * @param flag the flag to look up
   * @return the flag value
   * @throws IllegalArgumentException if the flag is absent
   */
  private static String required(Map<String, String> options, String flag) {
    String value = options.get(flag);
    if (value == null) {
      throw new IllegalArgumentException("Missing required option " + flag);
    }
    return value;
  }

  /**
   * Parses a positive integer option value.
   *
   * @param value the raw value
   * @param flag the flag name used in error messages
   * @return the parsed positive integer
   * @throws IllegalArgumentException if the value is not a positive integer
   */
  private static int positiveInt(String value, String flag) {
    try {
      int parsed = Integer.parseInt(value);
      if (parsed <= 0) {
        throw new NumberFormatException("not positive");
      }
      return parsed;
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(flag + " must be a positive integer but was '" + value + "'");
    }
  }

  /**
   * Parses a metric name, case-insensitively.
   *
   * @param value the raw metric name
   * @return the metric
   * @throws IllegalArgumentException if the name is not a known metric
   */
  private static Metric parseMetric(String value) {
    try {
      return Metric.valueOf(value.trim().toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException("Invalid --metric '" + value
          + "'; expected L2, COSINE, or INNER_PRODUCT");
    }
  }

  /**
   * Prints the usage message to standard error.
   */
  private static void printUsage() {
    System.err.println("Usage:");
    System.err.println("  hnswforge-eval groundtruth --base <fvecs> --queries <fvecs>"
        + " --metric L2|COSINE|INNER_PRODUCT --k N --out <ivecs>");
    System.err.println("  hnswforge-eval recall --truth <ivecs> --results <text> --k N");
    System.err.println("  hnswforge-eval sweep [--dataset sift1m|glove100|synthetic] [--n N]"
        + " [--m a,b,c] [--efc a,b,c] [--efs a,b,c] [--k N] [--threads N] [--out <csv>]");
  }

  /**
   * Reports a bad-argument failure with usage and the exit code two.
   *
   * @param message the error description
   * @return {@code 2}
   */
  private static int usageError(String message) {
    System.err.println("error: " + message);
    printUsage();
    return 2;
  }

  /**
   * Reports an I/O or data failure with the exit code one.
   *
   * @param message the error description
   * @return {@code 1}
   */
  private static int failure(String message) {
    System.err.println("error: " + message);
    return 1;
  }
}
