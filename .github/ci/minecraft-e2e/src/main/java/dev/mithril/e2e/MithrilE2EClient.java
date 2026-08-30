package dev.mithril.e2e;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL11;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Minimal, version-portable Minecraft client E2E probe for Mithril.
 *
 * The upstream lane drove the game through Fabric's client-gametest module.
 * That module does not exist for Minecraft 1.21.1, so this lane instead boots
 * the real production client to the main menu under a plain
 * {@code ClientModInitializer}, waits for a steady number of rendered ticks,
 * then samples the default framebuffer.
 *
 * The whole point of the sample is the red-screen regression: when every draw
 * sampled an undefined descriptor the entire surface came back saturated red
 * instead of the clear colour / menu content. Sampling a centred block and
 * measuring the fraction of "pure red" pixels turns that visual failure into a
 * hard, machine-checkable CI gate.
 */
public final class MithrilE2EClient implements ClientModInitializer {
    /** Ticks to wait so the loading screen / main menu is fully rendered. */
    private static final int WARMUP_TICKS = 120;
    /** Fraction of sampled pixels above which the screen counts as "pure red". */
    private static final double RED_FAIL_RATIO = 0.90;
    /** Hard deadline: if no sample has happened by now, kill the JVM anyway. */
    private static final long WATCHDOG_DEADLINE_MS = 420_000L;

    private int ticks = 0;
    private boolean done = false;
    /** Progress visible to the watchdog thread (and to its hang evidence). */
    private static final java.util.concurrent.atomic.AtomicInteger TICKS_SEEN =
            new java.util.concurrent.atomic.AtomicInteger(0);

    @Override
    public void onInitializeClient() {
        final Path root = Path.of(System.getProperty("mithril.e2e.root", "build/evidence"))
                .toAbsolutePath().normalize();

        // Watchdog: if the game never reaches the sample (stuck loading, blocked
        // render thread, or any other hang), nothing would ever call System.exit
        // and the CI step would burn its whole job timeout while the useful log
        // stayed unpublished. A daemon thread guarantees the JVM terminates on a
        // deadline so the evidence -- including the log showing where it hung --
        // is always published. halt() (not exit()) because a hung game may block
        // in a shutdown hook.
        Thread watchdog = new Thread(() -> {
            try {
                Thread.sleep(WATCHDOG_DEADLINE_MS);
            } catch (InterruptedException ignored) {
                return;
            }
            System.err.println("[mithril-e2e] WATCHDOG: deadline reached before the "
                    + "120-tick sample completed; forcing JVM halt");
            try {
                Files.createDirectories(root.resolve("render"));
                Map<String, Object> hang = new LinkedHashMap<>();
                hang.put("schema_version", "1.0");
                hang.put("reason", "watchdog deadline reached before sample");
                hang.put("deadline_ms", WATCHDOG_DEADLINE_MS);
                hang.put("ticks_observed", TICKS_SEEN.get());
                writeJson(root.resolve("hang-detected.json"), hang);
            } catch (Throwable t) {
                t.printStackTrace();
            }
            Runtime.getRuntime().halt(11);
        });
        watchdog.setDaemon(true);
        watchdog.setName("mithril-e2e-watchdog");
        watchdog.start();

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (done) return;
            ticks++;
            TICKS_SEEN.set(ticks);
            if (ticks < WARMUP_TICKS) return;
            done = true;

            try {
                Files.createDirectories(root.resolve("render"));

                String vendor = safe(GL11.glGetString(GL11.GL_VENDOR));
                String renderer = safe(GL11.glGetString(GL11.GL_RENDERER));
                String version = safe(GL11.glGetString(GL11.GL_VERSION));

                int w = client.getWindow().getWidth();
                int h = client.getWindow().getHeight();

                int sw = Math.max(1, Math.min(64, w));
                int sh = Math.max(1, Math.min(64, h));
                int x = (w - sw) / 2;
                int y = (h - sh) / 2;

                ByteBuffer px = BufferUtils.createByteBuffer(sw * sh * 4);
                GL11.glReadPixels(x, y, sw, sh, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, px);

                int total = sw * sh;
                int red = 0;
                for (int i = 0; i < total; i++) {
                    int r = px.get() & 0xFF;
                    int g = px.get() & 0xFF;
                    int b = px.get() & 0xFF;
                    px.get(); // alpha
                    if (r > 200 && g < 40 && b < 40) red++;
                }
                double redRatio = (double) red / (double) total;

                writeJson(root.resolve("game-state.json"), new LinkedHashMap<String, Object>() {{
                    put("schema_version", "1.0");
                    put("tick", ticks);
                    put("window_width", w);
                    put("window_height", h);
                    put("gl_vendor", vendor);
                    put("gl_renderer", renderer);
                    put("gl_version", version);
                    put("red_pixel_ratio", redRatio);
                    put("sampled_pixels", total);
                    put("red_pixels", red);
                }});

                boolean identityOk = version.contains("Mithril-Wrapper")
                        && renderer.contains("Mithril-Wrapper");
                boolean gameOk = w > 0 && h > 0;
                boolean renderOk = redRatio <= RED_FAIL_RATIO;

                writeJson(root.resolve("oracle-results.json"), new LinkedHashMap<String, Object>() {{
                    put("schema_version", "1.0");
                    put("l1_process", "pass");
                    put("l2_runtime_identity", identityOk ? "pass" : "fail");
                    put("l3_game_state", gameOk ? "pass" : "fail");
                    put("l4_gpu_render", renderOk ? "pass" : "fail");
                    put("l5_presentation", "diagnostic");
                }});

                System.out.println("[mithril-e2e] GL_VENDOR=" + vendor);
                System.out.println("[mithril-e2e] GL_RENDERER=" + renderer);
                System.out.println("[mithril-e2e] GL_VERSION=" + version);
                System.out.println("[mithril-e2e] red_pixel_ratio=" + redRatio);

                if (!renderOk) {
                    System.err.println("[mithril-e2e] FAILURE: pure-red screen detected "
                            + "(red_pixel_ratio=" + redRatio + ")");
                    writeJson(root.resolve("red-screen-detected.json"),
                            new LinkedHashMap<String, Object>() {{
                                put("schema_version", "1.0");
                                put("red_pixel_ratio", redRatio);
                            }});
                    System.exit(3);
                }
                System.exit(0);
            } catch (Throwable t) {
                t.printStackTrace();
                System.exit(4);
            }
        });
    }

    private static String safe(String s) { return s == null ? "" : s; }

    private static void writeJson(Path path, Map<String, Object> values) throws Exception {
        Files.createDirectories(path.getParent());
        StringBuilder b = new StringBuilder("{\n");
        int i = 0;
        for (Map.Entry<String, Object> e : values.entrySet()) {
            b.append("  \"").append(escape(e.getKey())).append("\":");
            Object v = e.getValue();
            if (v instanceof Number) {
                b.append(v instanceof Double || v instanceof Float
                        ? String.format(Locale.ROOT, "%.4f", ((Number) v).doubleValue())
                        : v.toString());
            } else if (v instanceof Boolean) {
                b.append(((Boolean) v) ? "true" : "false");
            } else {
                b.append('"').append(escape(String.valueOf(v))).append('"');
            }
            if (++i < values.size()) b.append(',');
            b.append('\n');
        }
        b.append("}\n");
        Files.writeString(path, b.toString(), StandardCharsets.UTF_8);
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' || c == '\\') sb.append('\\');
            sb.append(c);
        }
        return sb.toString();
    }
}
