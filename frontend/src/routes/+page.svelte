<script lang="ts">
    import UartService from "$lib/communication/UartService";

    const { bluetooth } = window.navigator;

    const constants = {
        handshakeMessage: "doodlebot",
        disconnectMessage: "disconnected",
        commandCompleteIdentifier: "done",
    };

    let popup: Window | null = null;

    let password: string = localStorage.getItem("password") || "";
    let ssid: string = localStorage.getItem("ssid") || "";
    let ip: string = localStorage.getItem("ip") || "";
    let playgroundURL: string = localStorage.getItem("playgroundURL") || "";

    $: if (password) localStorage.setItem("password", password);
    $: if (ssid) localStorage.setItem("ssid", ssid);
    $: if (ip) localStorage.setItem("ip", ip);
    $: if (playgroundURL) localStorage.setItem("playgroundURL", playgroundURL);

    $: popupOrigin = popup ? new URL(playgroundURL).origin : "";

    let unsubscribers = new Array<() => void>();

    const unsubscribe = () => {
        for (const unsubscribe of unsubscribers) unsubscribe();
        unsubscribers = [];
    };

    const forwardToPopup = ({
        detail,
    }: Pick<CustomEvent<string>, "detail">) => {
        popup?.postMessage(detail, playgroundURL);
    };

    const disconnect = () =>
        forwardToPopup({ detail: constants.disconnectMessage });

    const getUartService = async (
        service: BluetoothRemoteGATTService,
        device: BluetoothDevice,
    ) => {
        const uartService = await UartService.create(service);

        uartService.addEventListener("receiveText", forwardToPopup);
        unsubscribers.push(() =>
            uartService.removeEventListener("receiveText", forwardToPopup),
        );

        device.addEventListener("gattserverdisconnected", disconnect);
        unsubscribers.push(() =>
            device.removeEventListener("gattserverdisconnected", disconnect),
        );

        return uartService;
    };

    const waitForPopupToRespond = () =>
        new Promise<void>((resolve) => {
            let interval = setInterval(
                () =>
                    popup!.postMessage(
                        constants.handshakeMessage,
                        playgroundURL,
                    ),
                100,
            );

            const onReady = (event: MessageEvent) => {
                if (event.origin !== popupOrigin) return;
                clearInterval(interval!);
                window.removeEventListener("message", onReady);
                resolve();
            };

            window.addEventListener("message", onReady);
        });

    const connect = async () => {
        if (!bluetooth) return alert("Bluetooth not supported");

        unsubscribe();

        const device = await bluetooth.requestDevice({
            filters: [{ services: [UartService.uuid] }],
        });

        if (!device) return alert("No device selected");
        if (!device.gatt) return alert("No GATT server");

        if (!device.gatt.connected) await device.gatt.connect();

        const services = await device.gatt.getPrimaryServices();

        const found = services.find(
            (service) => service.uuid === UartService.uuid,
        );

        if (!found) return alert("No UART service found");

        const uartService = await getUartService(found, device);

        // handle case if playgroundURL ends in question mark (?)
        const url = new URL(playgroundURL);
        url.searchParams.set("password", password);
        url.searchParams.set("ssid", ssid);
        url.searchParams.set("ip", ip);

        popup = window.open(url.toString());

        if (!popup) return alert("Please allow popups for this website");

        await waitForPopupToRespond();

        const forwardToBLE = async ({ data, origin }: MessageEvent) => {
            if (origin !== popupOrigin) return;
            await uartService.sendText(data);
            forwardToPopup({
                detail: data + constants.commandCompleteIdentifier,
            });
        };

        window.addEventListener("message", forwardToBLE);
        unsubscribers.push(() =>
            window.removeEventListener("message", forwardToBLE),
        );
    };
</script>

<div class="row">
    <label for="ssid">ssid:</label>
    <input id="ssid" bind:value={ssid} class="fill" />
</div>

<div class="row">
    <label for="password">password:</label>
    <input id="password" bind:value={password} type="password" class="fill" />
</div>

<div class="row">
    <label for="ip">ip:</label>
    <input id="ip" bind:value={ip} class="fill" />
</div>

<div class="row">
    <label for="playgroundURL">playground URL:</label>
    <input id="playgroundURL" bind:value={playgroundURL} class="fill" />
</div>

<div>
    <button on:click={connect}>Connect</button>
</div>

<style>
    .row {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        gap: 1rem;
    }

    .fill {
        flex-grow: 1;
    }
</style>
