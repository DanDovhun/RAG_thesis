import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { AltoService } from './kramerius_alto_service';

const { window } = new JSDOM();
globalThis.DOMParser = window.DOMParser;

type Command = 'formatted' | 'fulltext' | 'boxes' | 'text-in-box';

interface CliOptions {
  uuid?: string;
  altoPath?: string;
  width: number;
  height: number;
  query?: string;
  box?: number[];
  useStdin: boolean;
}

interface ParsedArgs {
  command: Command;
  options: CliOptions;
}

function parseArgs(): ParsedArgs {
  const args = process.argv.slice(2);
  let command: Command = 'formatted';
  const opts: CliOptions = {
    width: 800,
    height: 1200,
    useStdin: false,
  };

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];

    if (arg === 'formatted' || arg === 'fulltext' || arg === 'boxes' || arg === 'text-in-box') {
      command = arg;
      continue;
    }

    if (arg.startsWith('--uuid=')) {
      opts.uuid = arg.split('=')[1];
      continue;
    }

    if (arg === '--uuid') {
      opts.uuid = args[++i];
      continue;
    }

    if (arg.startsWith('--alto=')) {
      opts.altoPath = arg.split('=')[1];
      continue;
    }

    if (arg === '--alto') {
      opts.altoPath = args[++i];
      continue;
    }

    if (arg.startsWith('--width=')) {
      opts.width = parseInt(arg.split('=')[1], 10);
      continue;
    }

    if (arg === '--width') {
      opts.width = parseInt(args[++i], 10);
      continue;
    }

    if (arg.startsWith('--height=')) {
      opts.height = parseInt(arg.split('=')[1], 10);
      continue;
    }

    if (arg === '--height') {
      opts.height = parseInt(args[++i], 10);
      continue;
    }

    if (arg.startsWith('--query=')) {
      opts.query = arg.split('=')[1];
      continue;
    }

    if (arg === '--query') {
      opts.query = args[++i];
      continue;
    }

    if (arg.startsWith('--box=')) {
      opts.box = JSON.parse(arg.split('=')[1]);
      continue;
    }

    if (arg === '--box') {
      opts.box = JSON.parse(args[++i]);
      continue;
    }

    if (arg === '--stdin') {
      opts.useStdin = true;
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  return { command, options: opts };
}

async function loadAlto(opts: CliOptions): Promise<string> {
  if (opts.useStdin) {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
    }
    return Buffer.concat(chunks).toString('utf-8');
  }

  if (opts.altoPath) {
    return readFileSync(opts.altoPath, 'utf-8');
  }

  if (!opts.uuid) {
    throw new Error('Missing UUID. Provide --uuid or --alto path.');
  }

  const url = `https://kramerius5.nkp.cz/search/api/v5.0/item/uuid:${opts.uuid}/streams/ALTO`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch ALTO: ${response.status} ${response.statusText}`);
  }

  return await response.text();
}

function ensureBox(box: CliOptions['box']): asserts box is number[] {
  if (!box || !Array.isArray(box)) {
    throw new Error('Missing --box argument with an array of numbers.');
  }
}

(async () => {
  try {
    const { command, options } = parseArgs();
    const alto = await loadAlto(options);
    const service = new AltoService();

    switch (command) {
      case 'formatted': {
        const output = service.getFormattedText(alto, options.uuid || '', options.width, options.height);
        console.log(output);
        break;
      }
      case 'fulltext': {
        const output = service.getFullText(alto);
        console.log(output);
        break;
      }
      case 'boxes': {
        if (!options.query) {
          throw new Error('Missing --query for boxes command.');
        }
        const output = service.getBoxes(alto, options.query, options.width, options.height);
        console.log(JSON.stringify(output));
        break;
      }
      case 'text-in-box': {
        ensureBox(options.box);
        const output = service.getTextInBox(alto, options.box, options.width, options.height);
        console.log(output);
        break;
      }
      default:
        throw new Error(`Unsupported command: ${command satisfies never}`);
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
})();
