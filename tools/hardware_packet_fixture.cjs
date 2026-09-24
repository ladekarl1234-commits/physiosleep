'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const packet = require('./hardware_packet_design.cjs');
const names = ['clean_plus','clean_minus','faults','signal_faults','missing_guard','missing_footer'];

function fixture(name) {
  if (!names.includes(name)) throw new Error('Unknown generated fixture');
  const ppm = name === 'clean_minus' ? -100 : 100;
  const first = 43138n * 250n, count = 15500;
  const offsetNs = 230000000n, slopeNs = 1000000000n + BigInt(ppm)*1000n;
  const referenceNs = counter => offsetNs + counter*slopeNs/250n;
  const originNs = referenceNs(first) + 1000700000n;
  const config = packet.fixtureConfig();
  config.capture_id = `generated-integration-${name}`;
  for (let ch = 0; ch < 4; ch++) Object.assign(config.channels[ch], {
    scale_uv_per_count: .01*(ch+1), offset_count: [.5,-3.25,7,-11][ch], uncertainty_uv: 0
  });
  const key = crypto.createHash('sha256').update(`PUBLIC-FIXTURE-KEY:${name}`).digest().subarray(0,16);
  const prefix = Buffer.from('1718191a1b1c1d1e','hex');
  const ctx = packet.context(config,key,prefix);
  const frames = [], mutations = [];
  for (let start = 0; start < count; start += 20) {
    const samples = [], statuses = [];
    for (let i = start; i < start+20; i++) {
      const t = Number(referenceNs(first+BigInt(i))-originNs)/1e9;
      const physical = [50*Math.sin(2*Math.PI*1.3*t+.2)+25*Math.sin(2*Math.PI*13*t-.1),
        40*Math.cos(2*Math.PI*.7*t+.3)+10*Math.sin(2*Math.PI*8.2*t+.6)];
      const row = [Math.round(physical[0]/.01+.5),Math.round(physical[1]/.02-3.25),123,-45];
      let status = 0xc00000;
      if (name === 'faults' || name === 'signal_faults') {
        if (i === 4200) row[0] = -8388608;
        if (i === 4300) row[1] = 8000001;
        if (i === 9500) status |= 0x001000;
        if (i === 9600) status = 0;
      }
      samples.push(row); statuses.push(status);
    }
    const sequence = start/20;
    let frame = packet.sealFixture(ctx, { samples, statuses, sequence,
      first:first+BigInt(start), tick:(first+BigInt(start))*4000n,
      flags:start+20 === count ? packet.FLAGS.FINAL : 0 });
    if (name === 'faults' && sequence === 200) {
      frame = Buffer.from(frame); frame[44] ^= 4; mutations.push('ciphertext byte44 of packet200');
    }
    if ((name === 'faults' && sequence === 388) || (name === 'missing_guard' && sequence >= 762)
        || (name === 'missing_footer' && sequence === 774)) {
      mutations.push(`omitted packet${sequence}`); continue;
    }
    frames.push(frame);
  }
  const assembled = packet.assemblePackets(frames,ctx,first,count);
  return { kind:'GENERATED_PACKET_CLOCK_FIXTURE_V1', name, node:process.version,
    openssl:process.versions.openssl, config:ctx.config, metadata_json:ctx.metadata_json,
    generator_clock:{offset_ns:offsetNs.toString(),slope_ppm:ppm},
    observation:{origin_ns:originNs.toString(),duration_ns:'60017000000',
      boundary_source:'independently_declared_synthetic_interval',fixed_before_fault_injection:true},
    synchronization:{event_error_seconds:0,exact_generated_events:true,
      counters:['0','5400000','10800000'],
      reference_ns:[0n,5400000n,10800000n].map(n => referenceNs(n).toString())},
    mutations, frames_base64:frames.map(x => x.toString('base64')), assembled };
}

module.exports = { fixture, names };
if (require.main === module) {
  if (process.argv[2] === '--verify') {
    try {
      const value = JSON.parse(fs.readFileSync(0,'utf8'));
      if (!names.includes(value.name)) throw new Error('Unknown fixture');
      const key = crypto.createHash('sha256').update(`PUBLIC-FIXTURE-KEY:${value.name}`).digest().subarray(0,16);
      const ctx = packet.context(value.config,key,Buffer.from('1718191a1b1c1d1e','hex'));
      assert.equal(ctx.metadata_json,value.metadata_json);
      const frames = value.frames_base64.map(x => Buffer.from(x,'base64'));
      const assembled = packet.assemblePackets(frames,ctx,43138n*250n,15500);
      assert.deepEqual(assembled,value.assembled);
      process.stdout.write(JSON.stringify({assembly_sha256:crypto.createHash('sha256')
        .update(JSON.stringify(assembled)).digest('hex')}));
    } catch {
      process.stderr.write('Generated packet replay disagrees with fixture assembly\n');
      process.exitCode = 1;
    }
  } else process.stdout.write(JSON.stringify(fixture(process.argv[2])));
}
