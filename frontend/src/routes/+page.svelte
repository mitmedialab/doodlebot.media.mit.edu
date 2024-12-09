<script lang="ts">
    import UartService from "$lib/communication/UartService";

    const { bluetooth } = window.navigator;

    const popupBaseURL = "http://doodlebot.media.mit.edu/playground";

    let popup: Window | null = null;

    let password: string;
    let ssid: string;
    let ip: string;

    let unsubscribers = new Array<() => void>();

    const unsubscribe = () => {
        for (const unsubscribe of unsubscribers) unsubscribe();
        unsubscribers = [];
    };

    const forwardToPopup = ({ detail }: Pick<CustomEvent<string>, "detail">) =>
        popup?.postMessage(detail, popupBaseURL);

    const disconnect = () => forwardToPopup({ detail: "disconnect" });

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
                () => popup!.postMessage("check", popupBaseURL),
                100,
            );

            const onReady = (event: MessageEvent) => {
                if (!event.origin.startsWith(popupBaseURL)) return;

                if (event.data === "ready") {
                    clearInterval(interval!);
                    window.removeEventListener("message", onReady);
                    resolve();
                }
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

        popup = window.open(
            `${popupBaseURL}?password=${password}&ssid=${ssid}&ip=${ip}`,
        );

        if (!popup) return alert("Please allow popups for this website");

        await waitForPopupToRespond();

        const forwardToBLE = async ({ data, origin }: MessageEvent) => {
            if (!origin.startsWith(popupBaseURL)) return;
            await uartService.sendText(data);
            forwardToPopup({ detail: data });
        };

        window.addEventListener("message", forwardToBLE);
        unsubscribers.push(() =>
            window.removeEventListener("message", forwardToBLE),
        );
    };
</script>

<label for="ssid">ssid:</label>
<input id="ssid" bind:value={ssid} />

<label for="password">password:</label>
<input id="password" bind:value={password} />

<label for="ip">ip:</label>
<input id="ip" bind:value={ip} />

<button on:click={connect}>Connect</button>
