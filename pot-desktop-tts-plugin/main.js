async function tts(text, _lang, options = {}) {
    const { config, utils } = options;
    const { http } = utils;
    const { fetch, Body } = http;

    let {
        serverUrl,
        exaggeration,
        cfgWeight,
        temperature,
        speed,
        pitch,
        speedAlgorithm,
        pitchAlgorithm,
        sampleRate
    } = config;

    // Build endpoint URL, defaulting to local FastAPI server; serverUrl should include full path/port
    let endpoint = (serverUrl || "http://localhost:8000/tts").trim();
    if (!/^https?:\/\//i.test(endpoint)) {
        endpoint = `http://${endpoint}`;
    }
    try {
        const url = new URL(endpoint);
        let path = url.pathname.replace(/\/+$/, "");
        if (!path.toLowerCase().endsWith("/tts")) {
            path = `${path}/tts`;
        }
        url.pathname = path;
        endpoint = url.toString().replace(/\/+$/, "");
    } catch (_e) {
        throw "Invalid server URL";
    }

    const payload = { text };

    const parsedExaggeration = parseFloat(exaggeration);
    if (!Number.isNaN(parsedExaggeration)) {
        payload.exaggeration = parsedExaggeration;
    }

    const parsedCfg = parseFloat(cfgWeight);
    if (!Number.isNaN(parsedCfg)) {
        payload.cfg_weight = parsedCfg;
    }

    const parsedTemp = parseFloat(temperature);
    if (!Number.isNaN(parsedTemp)) {
        payload.temperature = parsedTemp;
    }

    const parsedSpeed = parseFloat(speed);
    if (!Number.isNaN(parsedSpeed)) {
        payload.speed = parsedSpeed;
    }

    const parsedPitch = parseFloat(pitch);
    if (!Number.isNaN(parsedPitch)) {
        payload.pitch = parsedPitch;
    }

    if (speedAlgorithm && speedAlgorithm !== "auto") {
        payload.speed_algorithm = speedAlgorithm;
    }

    if (pitchAlgorithm && pitchAlgorithm !== "auto") {
        payload.pitch_algorithm = pitchAlgorithm;
    }

    const parsedSampleRate = parseInt(sampleRate, 10);
    if (!Number.isNaN(parsedSampleRate)) {
        payload.sample_rate = parsedSampleRate;
    }

    // Debug: log outgoing request target and payload
    console.log("[Chatterbox TTS] Endpoint:", endpoint);
    console.log("[Chatterbox TTS] Payload:", payload);

    const res = await fetch(endpoint, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: Body.json(payload),
        responseType: 3
    });

    if (res.ok) {
        const result = res.data;
        if (result) {
            const byteLength = result.length ?? result.byteLength ?? 0;
            console.log("[Chatterbox TTS] Received audio bytes:", byteLength);
            // Ensure we return a plain array for Pot's player
            if (result instanceof Uint8Array) {
                return Array.from(result);
            }
            if (Array.isArray(result)) {
                return result;
            }
            if (result instanceof ArrayBuffer) {
                return Array.from(new Uint8Array(result));
            }
            // Last resort: try to coerce to Uint8Array
            try {
                return Array.from(new Uint8Array(result));
            } catch (_e) {
                throw "Unexpected audio response format";
            }
        } else {
            throw "Empty audio response";
        }
    } else {
        let detail = "";
        try {
            detail = typeof res.data === "string" ? res.data : JSON.stringify(res.data);
        } catch (e) {
            detail = "";
        }
        throw `Http Request Error\nHttp Status: ${res.status}${detail ? `\n${detail}` : ""}`;
    }
}
