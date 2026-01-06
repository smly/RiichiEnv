"use strict";var RiichiEnvViewer=(()=>{var O=Object.defineProperty;var M=Object.getOwnPropertyDescriptor;var E=Object.getOwnPropertyNames;var R=Object.prototype.hasOwnProperty;var C=(y,n)=>{for(var e in n)O(y,e,{get:n[e],enumerable:!0})},L=(y,n,e,t)=>{if(n&&typeof n=="object"||typeof n=="function")for(let o of E(n))!R.call(y,o)&&o!==e&&O(y,o,{get:()=>n[o],enumerable:!(t=M(n,o))||t.enumerable});return y};var T=y=>L(O({},"__esModule",{value:!0}),y);var A={};C(A,{Viewer:()=>S});var G=y=>{let n=e=>{if(e==="back")return 9999;let t=e.slice(-1),o=parseInt(e[0]);(e.endsWith("r")||e.startsWith("0"))&&(o=5);let l={m:0,p:100,s:200,z:300},r=e.endsWith("r")||e.startsWith("0")?.1:0;return(l[t]||0)+o+r};return[...y].sort((e,t)=>n(e)-n(t))},_=class{constructor(n){this.events=n,this.cursor=0,this.current=this.initialState();let e=this.events.findIndex(t=>t.type==="start_kyoku");e!==-1?this.jumpTo(e+1):this.events.length>2&&this.jumpTo(2)}getKyokuCheckpoints(){let n=[];return this.events.forEach((e,t)=>{e.type==="start_kyoku"&&n.push({index:t,round:(e.kyoku||1)-1,honba:e.honba||0})}),n}initialState(){return{players:Array(4).fill(0).map(()=>({hand:[],discards:[],melds:[],score:25e3,riichi:!1,pendingRiichi:!1,wind:0})),doraMarkers:[],round:0,honba:0,kyotaku:0,wallRemaining:70,currentActor:0,eventIndex:0,totalEvents:this.events.length}}stepForward(){if(this.cursor>=this.events.length)return!1;let n=this.events[this.cursor];return this.processEvent(n),this.cursor++,this.current.eventIndex=this.cursor,!0}stepBackward(){if(this.cursor<=0)return!1;let n=this.cursor-1;for(this.reset();this.cursor<n;)this.stepForward();return!0}jumpTo(n){for(n<0&&(n=0),n>this.events.length&&(n=this.events.length),n<this.cursor&&this.reset();this.cursor<n;)this.stepForward()}stepTurn(n,e){let t=this.cursor,o=this.events.length;if(n)for(let l=this.cursor+1;l<o;l++){let w=this.events[l];if(w.actor===e&&["tsumo","dahai","pon","chi","kan","daiminkan","kakan","ankan"].includes(w.type)){t=l;break}}else for(let l=this.cursor-2;l>=0;l--){let w=this.events[l];if(w.actor===e&&["tsumo","dahai","pon","chi","kan","daiminkan","kakan","ankan"].includes(w.type)){t=l;break}}this.jumpTo(t+1)}reset(){this.cursor=0,this.current=this.initialState()}processEvent(n){switch(n.type){case"start_game":break;case"start_kyoku":this.current.round=(n.kyoku||1)-1,this.current.honba=n.honba||0,this.current.kyotaku=n.kyotaku||0,this.current.doraMarkers=[n.dora_marker],this.current.currentActor=n.oya,this.current.players.forEach((e,t)=>{e.hand=n.tehais[t].map(o=>o),e.discards=[],e.melds=[],e.riichi=!1,e.pendingRiichi=!1,e.score=n.scores[t],e.waits=void 0,e.wind=(t-n.oya+4)%4});break;case"tsumo":n.actor!==void 0&&n.pai&&(this.current.players[n.actor].hand.push(n.pai),this.current.currentActor=n.actor);break;case"dahai":if(n.actor!==void 0&&n.pai){let e=this.current.players[n.actor],t=e.hand.indexOf(n.pai);t>=0&&e.hand.splice(t,1),e.hand=G(e.hand);let o=!1;(e.pendingRiichi||n.reach)&&(o=!0,e.pendingRiichi=!1),e.discards.push({tile:n.pai,isRiichi:o,isTsumogiri:!!n.tsumogiri}),e.waits=n.meta?.waits,this.current.currentActor=n.actor}break;case"pon":case"chi":case"daiminkan":if(n.actor!==void 0&&n.target!==void 0&&n.pai&&n.consumed){let e=this.current.players[n.actor];n.consumed.forEach(o=>{let l=e.hand.indexOf(o);l>=0&&e.hand.splice(l,1)}),e.melds.push({type:n.type,tiles:[...n.consumed,n.pai],from:n.target}),e.waits=void 0,this.current.currentActor=n.actor;let t=this.current.players[n.target];if(t.discards.length>0){let o=t.discards.pop();o&&o.isRiichi&&(t.pendingRiichi=!0)}}break;case"kakan":case"ankan":if(n.actor!==void 0&&n.consumed){let e=this.current.players[n.actor];n.consumed.forEach(t=>{let o=e.hand.indexOf(t);o>=0&&e.hand.splice(o,1)}),e.melds.push({type:n.type,tiles:n.consumed,from:n.actor}),e.waits=void 0}break;case"reach":n.actor!==void 0&&(n.step==="1"&&(this.current.players[n.actor].pendingRiichi=!0),n.step==="2"&&(this.current.players[n.actor].riichi=!0,this.current.kyotaku+=1,this.current.players[n.actor].score-=1e3,this.current.players[n.actor].pendingRiichi=!1));break;case"dora":n.dora_marker&&this.current.doraMarkers.push(n.dora_marker);break;case"hora":case"ryukyoku":n.scores&&this.current.players.forEach((e,t)=>e.score=n.scores[t]);break}this.current.lastEvent=n}};var j={back:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Back.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Back.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
    <filter
       style="color-interpolation-filters:sRGB;"
       inkscape:label=""
       id="filter4198">
      <feGaussianBlur
         stdDeviation="2.51 2.51"
         result="blur"
         id="feGaussianBlur4200" />
    </filter>
    <mask
       maskUnits="userSpaceOnUse"
       id="mask4216">
      <rect
         ry="40"
         y="-1325.6035"
         x="-451.93805"
        
        
         id="rect4218"
         style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         transform="scale(-1,-1)" />
    </mask>
    <mask
       maskUnits="userSpaceOnUse"
       id="mask4222">
      <rect
         ry="40"
         y="652.28351"
         x="0"
        
        
         id="rect4224"
         style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
    </mask>
    <filter
       style="color-interpolation-filters:sRGB;"
       inkscape:label="Blur"
       id="filter4188">
      <feGaussianBlur
         stdDeviation="2.35 2.34"
         result="blur"
         id="feGaussianBlur4190" />
    </filter>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.35742971"
     inkscape:cx="-92.256393"
     inkscape:cy="-195.52606"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <rect
       style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="rect4164"
      
      
       x="0"
       y="652.28351"
       ry="40" />
    <path
       style="fill:#ffffff;fill-opacity:0.78431373;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;filter:url(#filter4198)"
       d="M -4.7687833,775.07096 C -9.6501835,741.99485 -16.84552,674.23676 -1.2788716,652.0641 18.998297,625.94378 233.50094,631.63117 263.31435,653.90999 276.21398,662.64856 59.158568,656.47658 35.908342,685.10013 13.199947,712.46776 0.65060847,818.18718 -4.7687833,775.07096 Z"
       id="path4166"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="ccccc"
       mask="url(#mask4222)" />
    <path
       sodipodi:nodetypes="ccccc"
       inkscape:connector-curvature="0"
       id="path4221"
       d="m 151.73588,1025.0177 c -3.32683,-9.3138 -10.24843,-68.45389 5.31821,-90.62655 20.27717,-26.12032 219.43558,-16.45796 231.55506,-9.93184 11.07433,5.31702 -178.60366,0.0589 -204.85126,34.86646 -21.59349,30.0006 -26.50086,82.17843 -32.02201,65.69193 z"
       style="fill:#000000;fill-opacity:0.39215687;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;filter:url(#filter4198)"
       transform="matrix(-1,0,0,-1,451.93806,1977.887)"
       mask="url(#mask4216)" />
  </g>
</svg>
`,blank:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Blank.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Blank.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="-66.998093"
     inkscape:cy="324.44382"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       style="font-style:normal;font-weight:normal;font-size:519.32067871px;line-height:125%;font-family:sans-serif;letter-spacing:0px;word-spacing:0px;fill:#d71e1e;fill-opacity:1;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       id="text11690">
      <path
         d="m 275.30302,756.03454 c 0,16.56687 -2.95837,31.35872 -8.87511,44.37555 -5.91674,12.84778 -13.69303,24.25863 -23.32886,34.23256 -9.46678,9.63583 -20.37048,18.67999 -32.71111,27.13248 -12.34063,8.45249 -25.44198,16.6514 -39.30406,24.59673 l 0,57.05428 -45.38984,0 0,-77.34024 c 10.98823,-6.25484 22.82171,-13.10135 35.50044,-20.53954 12.84777,-7.43819 23.32885,-14.9609 31.44324,-22.56813 9.80488,-8.79059 17.41212,-17.83475 22.82171,-27.13248 5.40959,-9.46679 8.11438,-21.46932 8.11438,-36.00759 0,-19.10261 -6.50841,-33.30279 -19.52524,-42.60052 -12.84777,-9.46679 -29.49917,-14.20018 -49.95418,-14.20018 -32.7283,1.20707 -79.344302,14.33075 -107.662353,40.62637 l -2.535745,0 0,-51.72921 c 31.218393,-21.8767 81.322938,-32.11693 113.748138,-32.2584 36.34569,0 64.99961,8.87511 85.96177,26.62532 21.13121,17.58117 31.69682,40.82551 31.69682,69.733 z m -100.16194,288.82146 -51.72921,0 0,-53.50427 51.72921,0 z"
         id="path5660"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="sccscccccccscccccccsccccc" />
    </g>
  </g>
</svg>
`,C:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Chun.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Chun.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="148.19657"
     inkscape:cy="260.89845"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g8162"
       transform="matrix(0.76410471,0,0,0.76410471,-168.70793,381.16585)">
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 233.73438,85.414062 c -62.01799,16.681888 -86.27161,21.909318 -166.158208,42.853518 -5.42522,-0.0965 -16.534496,-0.92756 -21.67167,-9.50781 -4.758196,-5.91895 -9.703104,0.68112 -9.437615,6.74218 0.365058,8.33409 0.598367,15.29187 4.480584,19.20117 21.664177,19.95035 29.078065,41.32218 32.222451,65.74024 -0.33085,5.21114 -4.62276,10.00283 3.193359,9.64844 7.869133,-0.58359 84.263689,-6.62945 140.414059,-1.69336 4.64178,0.47659 5.45942,-3.68138 5.15039,-5.39063 -2.64249,-14.6159 24.33847,-61.36206 33.375,-80.47461 1.86244,-3.9391 11.81804,-4.43434 13.9336,-4.93945 -0.13937,0.0348 2.86207,-1.72444 1.0957,-3.91406 C 259.613,110.39214 237.46714,84.235655 233.73438,85.414062 Z m -21.27735,34.902348 c 9.08657,0.0687 13.4201,2.43096 9.83008,8.34961 -11.11464,22.31009 -24.24336,65.93624 -28.21484,65.26367 -61.3062,-0.41953 -87.92695,4.0617 -94.01563,2.04687 -2.199134,-0.81536 -5.411834,-24.89934 -19.332875,-42.17976 -2.189357,-2.71769 -1.286435,-4.69532 2.253906,-5.55664 20.628929,-4.6163 99.796559,-28.14827 129.479359,-27.92375 z"
         transform="matrix(1.3087212,0,0,1.3087212,219.31497,352.30291)"
         id="path8156"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccsccccsscscscccscs" />
      <path
         sodipodi:nodetypes="cccccsccccc"
         inkscape:connector-curvature="0"
         id="path8160"
         d="m 367.90999,395.10475 c 0.37549,4.16112 -0.17568,12.40731 6.29079,14.78577 5.6162,0.58559 19.75278,11.44606 16.95438,28.57096 0.0928,187.92712 3.32312,200.16425 0.3131,345.45795 -5.62614,52.57681 6.73403,80.47048 10.82199,86.96513 2.93545,6.59024 6.12648,5.51697 7.60418,1.43859 12.97863,-35.82036 18.66091,-12.55418 14.70196,-102.90205 -0.58086,-150.13908 -2.88642,-315.64694 19.06631,-327.81061 8.67123,-6.96222 20.05495,-28.68846 17.34799,-33.72755 3.10315,-4.97486 -50.62215,-54.03954 -55.12546,-49.49693 -7.36818,8.03967 -36.64575,21.9603 -37.97524,36.71874 z"
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,front:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Front.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Front.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
    <filter
       style="color-interpolation-filters:sRGB;"
       inkscape:label="Blur"
       id="filter4198">
      <feGaussianBlur
         stdDeviation="2.51 2.51"
         result="blur"
         id="feGaussianBlur4200" />
    </filter>
    <mask
       maskUnits="userSpaceOnUse"
       id="mask4222">
      <rect
         ry="40"
         y="652.28351"
         x="0"
        
        
         id="rect4224"
         style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
    </mask>
    <filter
       style="color-interpolation-filters:sRGB"
       inkscape:label="Blur"
       id="filter4198-0">
      <feGaussianBlur
         stdDeviation="2.51 2.51"
         result="blur"
         id="feGaussianBlur4200-5" />
    </filter>
    <mask
       maskUnits="userSpaceOnUse"
       id="mask4216">
      <rect
         ry="40"
         y="-1325.6035"
         x="-451.93805"
        
        
         id="rect4218"
         style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         transform="scale(-1,-1)" />
    </mask>
    <filter
       style="color-interpolation-filters:sRGB"
       inkscape:label="Blur"
       id="filter4242">
      <feGaussianBlur
         stdDeviation="2.51 2.51"
         result="blur"
         id="feGaussianBlur4244" />
    </filter>
    <mask
       maskUnits="userSpaceOnUse"
       id="mask4222-0">
      <rect
         ry="40"
         y="652.28351"
         x="0"
        
        
         id="rect4224-8"
         style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
    </mask>
    <filter
       style="color-interpolation-filters:sRGB"
       inkscape:label="Blur"
       id="filter4198-8">
      <feGaussianBlur
         stdDeviation="2.51 2.51"
         result="blur"
         id="feGaussianBlur4200-3" />
    </filter>
    <mask
       maskUnits="userSpaceOnUse"
       id="mask4216-0">
      <rect
         ry="40"
         y="-1325.6035"
         x="-451.93805"
        
        
         id="rect4218-5"
         style="opacity:1;fill:#ff3737;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         transform="scale(-1,-1)" />
    </mask>
    <filter
       style="color-interpolation-filters:sRGB"
       inkscape:label="Blur"
       id="filter4242-2">
      <feGaussianBlur
         stdDeviation="2.51 2.51"
         result="blur"
         id="feGaussianBlur4244-2" />
    </filter>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.50548195"
     inkscape:cx="314.366"
     inkscape:cy="374.8318"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <rect
       style="opacity:1;fill:#f5f0eb;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="rect4164"
      
      
       x="0"
       y="652.28351"
       ry="40" />
    <path
       transform="translate(-1.3432789e-7,-1.3368765e-6)"
       style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;filter:url(#filter4198-8)"
       d="M -4.7687833,775.07096 C -9.6501835,741.99485 -16.84552,674.23676 -1.2788716,652.0641 18.998297,625.94378 233.50094,631.63117 263.31435,653.90999 276.21398,662.64856 70.349579,663.12124 47.099353,691.74479 24.390958,719.11242 0.65060847,818.18718 -4.7687833,775.07096 Z"
       id="path4166"
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="ccccc"
       mask="url(#mask4222-0)" />
    <path
       sodipodi:nodetypes="ccccc"
       inkscape:connector-curvature="0"
       id="path4221"
       d="m 151.73588,1025.0177 c -3.32683,-9.3138 -10.24843,-68.45389 5.31821,-90.62655 20.27717,-26.12032 219.43558,-16.45796 231.55506,-9.93184 11.07433,5.31702 -178.60366,0.0589 -204.85126,34.86646 -21.59349,30.0006 -26.50086,82.17843 -32.02201,65.69193 z"
       style="fill:#000000;fill-opacity:0.19607843;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;filter:url(#filter4198-8)"
       transform="matrix(-1,0,0,-1,451.93806,1977.887)"
       mask="url(#mask4216-0)" />
  </g>
</svg>
`,P:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Haku.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Haku.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="28.942359"
     inkscape:cy="260.89845"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)" />
</svg>
`,F:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Hatsu.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Hatsu.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="2.8594378"
     inkscape:cx="155.81431"
     inkscape:cy="284.12733"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g8214"
       transform="matrix(0.71142441,0,0,0.71142441,-140.4298,448.29298)">
      <path
         sodipodi:nodetypes="ccccccccc"
         inkscape:connector-curvature="0"
         id="path8167"
         d="m 409.81023,440.81379 c 34.62279,13.50198 72.53734,107.4651 100.96631,181.51977 1.39128,3.01271 2.23772,6.28074 5.69886,8.07115 25.98101,14.65951 56.13523,36.47186 68.69999,46.74349 23.44422,16.60038 35.69055,5.67982 29.30548,-10.11988 -4.82392,-12.17183 -23.57808,-40.28999 -27.77314,-49.28727 -5.34853,-9.97789 -19.28999,-21.42322 -37.2864,-34.99872 -35.85489,-25.86472 -57.62913,-82.48687 -122.2875,-162.60639 -3.15221,-2.99045 -19.00814,14.44305 -17.3236,20.67785 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path8169"
         d="m 423.61741,432.98337 c -78.38452,110.34146 -127.9378,173.34666 -203.96634,220.50138 -21.02224,14.3026 -25.41866,2.34112 -9.19239,-10.6066 93.49668,-76.12907 113.84213,-140.11107 192.05269,-224.6799 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path8171"
         d="m 401.08262,388.1417 c -30.0844,27.27543 -48.12573,58.97041 -85.14675,77.97873 -16.65631,10.41567 -17.18931,22.16526 4.09314,19.50193 34.74776,-5.57046 73.28108,-26.55584 101.8108,-43.31264 24.08363,-14.96831 21.09071,-89.46719 -20.75719,-54.16802 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccscc"
         inkscape:connector-curvature="0"
         id="path8173"
         d="m 473.08767,378.62869 c -8.02066,26.87021 -15.13164,47.69091 -31.25245,75.39126 l 18.53038,17.01419 c 25.67346,-51.40051 34.47825,-76.25865 47.31398,-80.03288 4.27196,-2.31981 1.63306,-10.04729 -0.56497,-14.07335 -4.61276,-8.44904 -9.25981,-18.30206 -15.06625,-24.31164 -12.59872,-12.06953 -14.78517,14.81208 -18.96069,26.01242 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccc"
         inkscape:connector-curvature="0"
         id="path8175"
         d="m 525.86584,454.91235 c -5.54071,-6.01002 -11.79755,-3.89243 -16.34237,6.76653 -4.08052,9.96218 -25.34737,37.03766 -51.45459,62.21686 -8.90795,9.08189 -0.99657,17.81044 13.05438,7.92798 23.50888,-22.22451 42.38039,-39.49532 56.80839,-42.84371 6.5576,-1.2916 18.03663,2.91589 16.74839,-6.01028 -2.32799,-10.73573 -9.92204,-19.25567 -18.8142,-28.05738 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccsc"
         inkscape:connector-curvature="0"
         id="path8177"
         d="m 291.14056,495.74433 c -18.98477,-18.22387 -34.16175,-11.58186 -14.20356,15.6529 41.32415,56.90934 66.67472,67.7742 56.66777,88.65182 l 19.17085,0.006 c 10.56156,-11.40712 6.78136,-4.01512 -0.95764,-38.69447 -2.20601,-9.8854 -31.08882,-24.82546 -60.67742,-65.61603 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path8179"
         d="m 312.31994,554.81732 -27.33931,26.17205 c 34.68706,-2.84292 50.37048,-0.45207 85.66626,-20.67923 24.81809,-13.68241 9.35868,-25.97795 -4.46845,-21.24951 -29.98684,12.62584 -28.04303,11.23429 -53.8585,15.75669 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccc"
         inkscape:connector-curvature="0"
         id="path8181"
         d="m 278.30827,635.42474 c -7.82525,4.63886 -17.68122,-3.49023 -12.31478,8.80297 5.32744,9.29246 16.15426,8.27354 26.71738,4.40094 9.13575,-3.38913 71.71491,-51.79874 112.07759,-57.50707 l 2.14321,-27.02719 c -65.50803,22.159 -120.90979,66.04201 -128.6234,71.33035 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccc"
         inkscape:connector-curvature="0"
         id="path8183"
         d="m 426.86553,485.78155 c -3.19076,-3.71578 -11.41929,-9.25124 -15.63593,4.07965 -0.96026,6.26721 1.53684,12.49406 0.64722,22.12513 -4.21159,57.88331 -16.97927,75.4203 -21.17891,99.26947 -1.93549,12.71196 3.74737,21.5917 12.91784,3.33741 6.85173,-20.53521 14.51284,-35.09638 37.88187,-94.81878 1.55601,-10.54656 -4.22291,-22.31383 -14.63209,-33.99288 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccccsc"
         inkscape:connector-curvature="0"
         id="path8185"
         d="m 352.08956,608.30937 -15.28641,8.07427 c 9.5478,22.70367 -13.1565,103.21785 -30.12907,101.91066 -20.39639,-2.1256 -40.39833,-17.08742 -55.56712,-28.00061 l -6.8703,0.63016 c 16.88066,27.58559 39.19792,59.94684 63.6674,64.40556 7.7044,1.21915 22.74713,-8.45442 26.73484,-15.05843 29.58096,-60.78744 39.33111,-91.75066 33.12844,-109.08526 -2.84264,-7.94434 -12.71671,-13.94044 -15.67778,-22.87635 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccccc"
         inkscape:connector-curvature="0"
         id="path8187"
         d="m 344.58979,659.10347 -2.1022,-18.39532 c -37.05673,12.70616 -65.78042,33.73461 -94.64108,38.10725 -11.4765,1.53953 -20.1626,-2.30626 -28.06427,-7.63271 -5.32155,-4.2569 -7.88849,-0.63662 -6.81749,4.92746 2.99294,11.73226 4.71481,11.44031 12.11058,17.20458 6.91391,4.25918 12.47336,4.55663 23.63769,3.78598 28.29404,-2.2201 64.48105,-27.74468 95.87677,-37.99724 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path8189"
         d="m 404.43319,591.78934 -8.73076,26.94862 c 32.17108,-4.28365 49.87113,-6.04601 78.51848,-16.54145 22.25023,-16.457 10.3826,-20.90654 -3.40055,-22.33296 -39.38769,3.2834 -26.07867,6.79763 -66.38717,11.92579 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccc"
         inkscape:connector-curvature="0"
         id="path8191"
         d="m 441.25038,600.19989 -21.24999,5.98177 c -0.493,41.25671 0.96768,49.10319 -3.35007,59.74705 -13.11619,25.02851 -2.41912,29.6405 -63.80527,75.31466 -8.5931,6.22217 -0.71461,18.32446 9.28234,16.44145 42.48216,-7.09665 79.34052,-32.53701 81.81257,-42.05122 5.78926,-28.21841 0.60878,-70.23137 -2.68958,-115.43371 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccc"
         inkscape:connector-curvature="0"
         id="path8193"
         d="m 355.04551,652.39514 -1.24,24.69928 c -5.02581,-1.64396 99.9555,-7.84793 163.56733,-3.18873 8.38786,-0.31554 20.89211,0.17295 18.95537,-7.29219 -14.30223,-21.13914 -22.1051,-23.50845 -30.75289,-24.52112 -2.9899,0.26289 -8.58606,3.05248 -12.6085,2.70199 -25.40401,2.1586 -87.69427,6.19249 -137.92131,7.60077 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cscccc"
         inkscape:connector-curvature="0"
         id="path8195"
         d="m 467.46975,707.99594 c 16.45636,11.80939 23.18013,23.60264 27.44222,47.64951 0.48027,2.70969 -0.53953,7.36472 7.27341,8.72373 28.71731,4.99955 57.74678,-8.73379 60.29802,-22.09624 1.59916,-27.28194 -51.93367,-50.90188 -89.27628,-47.74791 -7.69401,1.28961 -9.12776,11.01879 -5.73737,13.47091 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"1m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man1.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man1.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="-228.09521"
     inkscape:cy="266.37243"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4171">
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 52.894354,722.32216 c -17.50696,-0.11271 -9.05357,18.89889 -0.90638,19.47161 56.421096,12.31796 92.339806,-11.63681 131.424976,-9.52501 20.67898,2.05086 53.82412,11.04089 61.51362,15.63063 1.63798,0.97768 3.97814,0.74093 4.99547,0.20017 8.33935,-4.39359 7.71325,-25.10483 4.35766,-36.85299 -0.93149,-3.48692 -8.66932,-7.71099 -15.05624,0.17833 -2.0941,2.11098 -10.8512,0.64005 -12.70543,-0.6243 -8.37862,-5.71316 -31.44441,-11.18628 -43.55827,-8.08264 -46.22841,9.75766 -82.7463,26.55006 -130.065406,19.6042 z"
         id="path5527"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccscccscc" />
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
    </g>
  </g>
</svg>
`,"2m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man2.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man2.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="-40.64517"
     inkscape:cy="271.06363"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4173">
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
      <g
         transform="translate(4.91793,1.7e-5)"
         id="g5576">
        <path
           sodipodi:nodetypes="ccccccc"
           inkscape:connector-curvature="0"
           id="path5578"
           d="m 54.10195,754.95801 c -15.912011,-3.454 -14.930316,14.04117 -0.938626,17.70218 58.428346,18.96345 99.345046,-16.31334 139.820716,-15.4839 20.30987,1.1749 35.04555,16.99112 53.7222,14.62781 34.98213,-7.283 16.1489,-30.75749 -1.64636,-33.70036 -14.28494,-2.45876 -27.33316,-7.5262 -52.90732,-3.77106 -49.69642,9.55913 -98.75938,31.71943 -138.05061,20.62533 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 93.901577,699.2551 c -6.43699,1.43264 -9.591843,15.77707 0.387269,17.60976 34.313284,-0.0935 73.545784,-13.35894 112.423824,-18.66638 6.63976,-1.36439 7.75026,-14.50373 8.19395,-19.04081 1.47048,-7.12488 -3.42052,-9.17959 -9.24235,-7.88429 -38.85688,9.78523 -72.90583,22.04384 -111.762693,27.98172 z"
           id="path5580"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccc" />
      </g>
    </g>
  </g>
</svg>
`,"3m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man3.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man3.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.71485945"
     inkscape:cx="48.858905"
     inkscape:cy="343.50159"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4253">
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
      <g
         transform="translate(-0.31322,0)"
         id="g5622">
        <path
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 62.315839,776.95528 c -15.657846,0.14446 -14.691832,16.45959 -0.923632,17.70218 58.869103,1.54525 97.758173,-27.33279 137.587323,-22.02469 19.98546,1.1749 39.02169,17.27551 57.40002,14.91219 9.42656,-7.96723 7.5453,-30.35197 -1.24208,-31.70969 -49.95288,-9.82532 -27.62306,-12.21431 -56.97614,-6.04611 -48.90261,9.55913 -97.1292,27.16612 -135.845491,27.16612 z"
           id="path5624"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="ccccccc" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path5626"
           d="m 89.661002,684.20854 c -6.91223,1.5492 -10.30002,17.06074 0.41586,19.04255 36.846648,6.14772 78.975718,-10.70797 120.724168,-20.18515 7.12998,-1.4754 10.94035,-13.12436 11.4168,-18.03059 1.57904,-7.70459 -6.29095,-12.48591 -12.5426,-11.08522 -41.00285,16.91408 -78.10031,31.2545 -120.014228,30.25841 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 89.841656,728.57665 c -6.115829,1.42706 -9.113286,15.71566 0.367952,17.54122 32.601292,5.66302 74.422712,-9.86375 111.361042,-18.59373 6.30848,-1.35909 7.36358,-14.44728 7.78513,-18.9667 1.39711,-7.09715 -3.24986,-9.14386 -8.78122,-7.85359 -36.27862,15.58055 -73.64819,28.79036 -110.732904,27.8728 z"
           id="path5628"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccc" />
      </g>
    </g>
  </g>
</svg>
`,"4m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man4.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man4.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="50.488415"
     inkscape:cy="288.48589"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4230">
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
      <g
         id="g13338"
         transform="matrix(1.0799971,0,0,0.95914962,-11.6365,29.661665)">
        <path
           sodipodi:nodetypes="sccccccccccsccccccccc"
           inkscape:connector-curvature="0"
           id="path13340"
           d="m 196.19505,673.02473 c -9.70415,0.0945 -20.98362,1.03933 -34.2754,3.08984 -29.38399,5.48615 -80.089202,30.0434 -95.829882,22.55177 -6.567867,-3.3436 -14.090403,-8.27415 -20.985677,-12.27256 -8.729299,-5.37629 -14.17276,-2.69115 -9.791015,9.30469 l 57.799938,75.62605 c 3.70119,6.32026 10.899816,5.21258 16.349606,3.75977 38.28834,-15.07486 61.25773,-10.00154 93.05676,0.3823 7.35538,2.22347 10.83485,0.7596 16.22976,-3.87925 18.04071,-17.1987 43.53355,-41.93845 43.99278,-49.60753 0.55707,-3.54122 3.24357,-24.4691 -2.90039,-29.08008 -14.41978,-9.70663 -28.98883,-20.21243 -63.64648,-19.875 z m 31.33789,29.87304 c 4.08988,3.92198 3.09909,9.22014 0.34961,13.98829 -7.24574,13.14893 -13.56284,23.15248 -25.93176,34.46169 -2.13947,1.74062 -4.25728,3.52934 -6.6327,3.65745 -14.6062,-1.95055 -5.84969,-12.66022 -76.9375,1.36914 -4.84299,0.0955 -6.99204,-0.94675 -8.74414,-2.79688 L 82.53294,715.28254 c -2.633261,-5.43934 2.487319,-4.52388 5.771484,-4.54688 39.937986,0.11061 96.302226,-40.71123 139.228516,-7.83789 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path13342"
           d="m 119.85989,691.9388 c 2.81402,21.7202 3.57454,43.44039 3.9963,65.16059 l 15.04843,-0.49458 c 14.54041,-23.12051 8.78886,-45.48553 5.44946,-68.62263 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path13344"
           d="m 193.61653,674.85102 c -4.14094,-18.99273 -21.21134,-9.06923 -23.27679,-0.597 l -12.56525,80.37248 15.70606,0.98916 c 10.27333,-28.16682 22.45499,-64.53631 27.78763,-71.46645 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
    </g>
  </g>
</svg>
`,"5mr":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man5-Dora.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Man5-Dora.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="179.88567"
     inkscape:cy="252.8135"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g6031"
       transform="translate(1034.3429,648.88567)">
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
         id="path6014"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
         id="path6017"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="sccccccs" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
         id="path6019"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="csccccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
         id="path6021"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccscccccccccsccccccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
         id="path6023"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
         id="path6025"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
         id="path6027"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
         id="path6029"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccccccccccccc" />
    </g>
    <g
       transform="matrix(0.88299048,0,0,0.98816906,13.1912,3.648797)"
       id="g5716">
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 120.27793,677.75471 c -7.8385,4.68172 -12.0636,5.88462 -15.28389,15.76934 -0.51069,1.5984 -0.89802,7.87579 1.45828,12.18189 20.42172,17.87538 -53.387678,56.71165 -79.77403,73.05807 -17.1742039,10.05844 -11.334569,21.33928 11.1927,11.45291 29.066321,-12.07831 80.29683,-26.14346 90.36555,-62.50547 0.0974,-3.98883 3.07469,-5.6683 7.33092,-6.61801 6.5143,-1.42607 0.79247,-27.81439 -2.95115,-34.69169 -3.74364,-6.87729 -9.80421,-9.95916 -12.33838,-8.64704 z"
         id="path5718"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 99.312378,743.71137 c 1.356942,4.63869 6.087332,16.55244 5.223812,21.54646 -3.97785,9.12562 -9.539572,12.59405 -11.761299,22.30502 -0.951932,5.34243 18.599519,21.85854 24.214479,19.2474 7.17233,-4.34808 10.98357,-13.34363 10.86546,-17.38476 -0.77831,-19.43426 -13.16868,-33.19626 -13.6164,-55.56915 z"
         id="path5720"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 128.52177,684.93516 c 46.74633,15.51693 66.83889,-16.27548 109.09748,-24.48481 12.11701,-2.3539 24.18389,2.32491 33.4817,4.16336 20.10867,4.51419 4.22782,31.73865 -11.12198,25.40944 -36.55813,-16.13394 -67.90561,8.15603 -102.60538,15.75123 -2.56615,0.41461 -4.13405,-2.70999 -5.7732,-4.83223 -5.31005,-6.87503 -13.32357,-1.20308 -19.85872,-3.34769 z"
         id="path5722"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cscccscc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 183.56224,691.9042 -31.76516,6.91681 c 9.85021,10.81876 -13.47538,53.48216 -25.20301,81.45969 l 28.50682,-3.84691 c 2.86369,-29.1146 21.0972,-56.08602 28.46135,-84.52959 z"
         id="path5724"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 114.44574,721.80311 c 15.7995,34.88788 69.14952,-14.0007 103.23262,-11.18105 l 0,19.4218 c -34.97155,-5.66636 -93.65276,35.5377 -107.63487,6.71249 z"
         id="path5726"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 210.5717,701.26372 c -2.78547,-0.46571 -10.74507,7.25181 -10.72639,10.04644 0.68906,24.48817 0.0838,48.42955 -2.14675,72.91772 -0.80858,11.46836 23.90304,2.38393 24.92498,-3.04856 3.43539,-25.63932 1.95788,-74.42799 -0.25808,-75.63801 -3.42035,-2.01357 -6.72578,-3.3576 -11.79376,-4.27759 z"
         id="path5728"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 112.95927,778.81549 c 109.31862,-31.08806 166.20953,-8.22783 169.39263,-2.25441 2.14803,3.27773 1.68381,10.10921 -0.57407,13.53111 -18.85836,25.07686 -119.62667,-39.36712 -165.06741,10.70831 z"
         id="path5730"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
    </g>
    <circle
       style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="path4178"
       cx="53.414371"
       cy="720.5152"
       r="19.7831" />
  </g>
</svg>
`,"5m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man5.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man5.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="126.09069"
     inkscape:cy="277.10934"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g6031"
       transform="translate(1034.3429,648.88567)">
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
         id="path6014"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
         id="path6017"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="sccccccs" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
         id="path6019"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="csccccccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
         id="path6021"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccscccccccccsccccccccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
         id="path6023"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
         id="path6025"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
         id="path6027"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
         id="path6029"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccccccccccccc" />
    </g>
    <g
       transform="matrix(0.88299048,0,0,0.98816906,13.1912,3.648797)"
       id="g5716">
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 120.27793,677.75471 c -7.8385,4.68172 -12.0636,5.88462 -15.28389,15.76934 -0.51069,1.5984 -0.89802,7.87579 1.45828,12.18189 20.42172,17.87538 -53.387678,56.71165 -79.77403,73.05807 -17.1742039,10.05844 -11.334569,21.33928 11.1927,11.45291 29.066321,-12.07831 80.29683,-26.14346 90.36555,-62.50547 0.0974,-3.98883 3.07469,-5.6683 7.33092,-6.61801 6.5143,-1.42607 0.79247,-27.81439 -2.95115,-34.69169 -3.74364,-6.87729 -9.80421,-9.95916 -12.33838,-8.64704 z"
         id="path5718"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 99.312378,743.71137 c 1.356942,4.63869 6.087332,16.55244 5.223812,21.54646 -3.97785,9.12562 -9.539572,12.59405 -11.761299,22.30502 -0.951932,5.34243 18.599519,21.85854 24.214479,19.2474 7.17233,-4.34808 10.98357,-13.34363 10.86546,-17.38476 -0.77831,-19.43426 -13.16868,-33.19626 -13.6164,-55.56915 z"
         id="path5720"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 128.52177,684.93516 c 46.74633,15.51693 66.83889,-16.27548 109.09748,-24.48481 12.11701,-2.3539 24.18389,2.32491 33.4817,4.16336 20.10867,4.51419 4.22782,31.73865 -11.12198,25.40944 -36.55813,-16.13394 -67.90561,8.15603 -102.60538,15.75123 -2.56615,0.41461 -4.13405,-2.70999 -5.7732,-4.83223 -5.31005,-6.87503 -13.32357,-1.20308 -19.85872,-3.34769 z"
         id="path5722"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cscccscc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 183.56224,691.9042 -31.76516,6.91681 c 9.85021,10.81876 -13.47538,53.48216 -25.20301,81.45969 l 28.50682,-3.84691 c 2.86369,-29.1146 21.0972,-56.08602 28.46135,-84.52959 z"
         id="path5724"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 114.44574,721.80311 c 15.7995,34.88788 69.14952,-14.0007 103.23262,-11.18105 l 0,19.4218 c -34.97155,-5.66636 -93.65276,35.5377 -107.63487,6.71249 z"
         id="path5726"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 210.5717,701.26372 c -2.78547,-0.46571 -10.74507,7.25181 -10.72639,10.04644 0.68906,24.48817 0.0838,48.42955 -2.14675,72.91772 -0.80858,11.46836 23.90304,2.38393 24.92498,-3.04856 3.43539,-25.63932 1.95788,-74.42799 -0.25808,-75.63801 -3.42035,-2.01357 -6.72578,-3.3576 -11.79376,-4.27759 z"
         id="path5728"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 112.95927,778.81549 c 109.31862,-31.08806 166.20953,-8.22783 169.39263,-2.25441 2.14803,3.27773 1.68381,10.10921 -0.57407,13.53111 -18.85836,25.07686 -119.62667,-39.36712 -165.06741,10.70831 z"
         id="path5730"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
    </g>
  </g>
</svg>
`,"6m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man6.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man6.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="112.14056"
     inkscape:cy="223.8092"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4226">
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
      <g
         id="g4220">
        <path
           sodipodi:nodetypes="cccccccscc"
           inkscape:connector-curvature="0"
           id="path13466"
           d="m 138.7313,661.52206 c -6.6616,-6.95081 -9.1444,-5.77088 -14.0104,-1.95112 -3.9,3.99773 -7.6141,6.22696 -9.4029,14.14555 -0.9672,4.35856 -0.1357,13.04214 1.9511,20.48664 1.2872,3.66163 6.0191,7.38107 11.0053,8.77999 1.7089,0.54178 -0.6227,8.67159 -2.4389,13.00739 l 18.2103,-0.97556 c -1.542,-4.90025 -5.4918,-12.83803 -4.5526,-14.30813 5.0996,-7.98175 10.8344,-18.72384 10.8605,-22.76293 0.07,-3.68127 -5.6263,-10.16306 -11.6224,-16.42183 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccccc"
           inkscape:connector-curvature="0"
           id="path13468"
           d="m 64.5497,730.18985 c -11.5923,-1.20004 -23.5395,-15.84843 3.0287,-14.30813 42.5006,3.24585 75.1937,-0.24655 131.6891,-25.61069 6.9088,-1.68598 58.3245,12.34434 60.4325,17.55997 2.6536,7.90859 -1.0286,30.33113 -12.6659,29.49377 -16.4723,-5.20503 -53.4279,-24.45551 -66.3743,-20.98316 -49.4555,12.15852 -66.6379,17.11767 -116.1101,13.84824 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccc"
           inkscape:connector-curvature="0"
           id="path13470"
           d="m 133.8713,733.00545 c -4.023,-2.1393 -24.469,13.50502 -24.8969,16.38959 -0.8458,5.32517 4.7863,12.9199 2.2994,14.18942 -35.1381,18.20437 -47.3832,19.28361 -65.468,22.48717 -21.4243,4.19792 -16.0723,15.92019 0,14.16461 50.8401,-5.29518 39.7211,-11.78632 92.1741,-21.98182 4.0787,-0.89675 11.8687,-18.05755 11.8687,-24.79233 0,-5.72987 -10.2767,-17.43517 -15.9773,-20.45664 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccc"
           inkscape:connector-curvature="0"
           id="path13472"
           d="m 160.5072,742.24577 c -2.0319,-0.11794 -2.5658,0.66243 -2.0807,1.81406 10.9073,25.89761 58.7952,61.45832 61.3263,56.12107 7.4362,-17.11962 5.1809,-19.57145 2.0287,-24.69006 -11.8554,-19.09585 -43.2116,-31.98957 -61.2743,-33.24507 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
    </g>
  </g>
</svg>
`,"7m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man7.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man7.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="2.0219279"
     inkscape:cx="70.788938"
     inkscape:cy="271.38275"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g6031"
       transform="translate(1034.3429,648.88567)">
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
         id="path6014"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
         id="path6017"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="sccccccs" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
         id="path6019"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="csccccccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
         id="path6021"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccscccccccccsccccccccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
         id="path6023"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
         id="path6025"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
         id="path6027"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
         id="path6029"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccccccccccccc" />
    </g>
    <g
       transform="matrix(1.1103116,0,0,1,-20.0629,0)"
       id="g13512">
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 122.12365,655.71644 c -9.04375,-1.38864 -18.45494,18.7911 -19.0105,23.98575 1.47894,11.46445 10.50381,15.44155 11.95913,20.17827 6.50446,25.75105 8.4039,67.28056 10.49584,75.3199 2.20074,8.02132 55.9263,13.84495 59.72476,11.7353 11.84665,-8.00981 17.24866,-31.32164 6.85982,-29.21397 -21.90078,3.13098 -39.53972,3.41801 -44.17005,0.63215 -3.97158,-2.69778 -12.89187,-48.4919 -10.86615,-52.66272 2.48997,-5.51509 11.95588,-12.68542 10.81645,-17.48596 -1.03414,-5.10364 -15.0153,-30.79118 -25.8093,-32.48872 z"
         id="path13514"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccccccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="M 71.063411,776.82976 C 59.009706,777.70468 60.345873,760.83508 70.622258,758.0431 209.49254,737.5656 108.95092,704.0102 237.90225,682.07575 c 5.62581,-1.10813 4.97456,27.02373 -0.70918,27.69935 -129.99354,18.71741 -22.68823,40.7935 -166.129659,67.05466 z"
         id="path13516"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
    </g>
  </g>
</svg>
`,"8m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man8.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man8.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.50548198"
     inkscape:cx="171.14265"
     inkscape:cy="553.68565"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4212">
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
      <g
         id="g5817"
         transform="matrix(0.92702391,0,0,1.0503994,15.0386,-40.624854)">
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path5819"
           d="m 114.0761,734.8528 c -5.28616,-0.85407 -14.805908,18.76719 -18.704926,21.30127 -22.082335,14.12373 -39.340383,15.36743 -63.567255,21.64549 -12.721217,3.36603 -11.360075,14.21856 3.418439,11.8164 24.396476,-4.90767 96.499612,-10.34096 102.983592,-36.95124 2.05404,-10.19543 -13.81066,-15.1829 -24.12985,-17.81192 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccccc"
           inkscape:connector-curvature="0"
           id="path5821"
           d="m 124.63085,691.12602 c 1.76248,-4.83916 16.21347,-12.94701 26.26906,-14.50456 10.20046,-0.98621 29.49486,43.70883 36.95666,50.26387 14.31491,11.39763 91.83537,46.43368 86.4741,63.54135 -3.2343,9.64522 -61.69009,-17.15445 -92.22719,-31.81371 C 164.95619,747.83945 151.50266,729.48888 142.98717,707.86691 141.4678,703.089 122.774,695.70076 124.63085,691.12602 Z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
    </g>
  </g>
</svg>
`,"9m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man9.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Man9.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="110.92188"
     inkscape:cy="343.20912"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4231">
      <g
         transform="translate(1034.3429,648.88567)"
         id="g6031">
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6014"
           d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="sccccccs"
           inkscape:connector-curvature="0"
           id="path6017"
           d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="csccccccccc"
           inkscape:connector-curvature="0"
           id="path6019"
           d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccscccccccccsccccccccccc"
           inkscape:connector-curvature="0"
           id="path6021"
           d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccc"
           inkscape:connector-curvature="0"
           id="path6023"
           d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6025"
           d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccccc"
           inkscape:connector-curvature="0"
           id="path6027"
           d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="cccccccccccccccc"
           inkscape:connector-curvature="0"
           id="path6029"
           d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
           style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
      <g
         transform="translate(-0.3219,0)"
         id="g5482">
        <path
           sodipodi:nodetypes="cccsc"
           inkscape:connector-curvature="0"
           id="path5484"
           d="m 146.54838,677.43662 c 0.54434,-9.2727 -36.81274,-17.9586 -33.24598,-4.27126 17.95022,62.01984 -5.54222,79.38907 -66.76663,100.84189 -4.171059,1.14614 -5.268439,9.32072 1.845649,7.11565 C 112.9187,761.11914 152.22385,756.52046 146.54838,677.43662 Z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccssscccc"
           inkscape:connector-curvature="0"
           id="path5486"
           d="m 121.57594,718.65069 c -5.84347,-2.20134 -10.97704,-5.4517 -14.36291,-10.56747 -0.6003,-0.89938 -2.22977,-8.68579 1.97466,-9.14854 16.17404,-1.78017 7.17997,-6.60782 10.67327,-7.33243 21.63049,-4.4868 52.3212,-17.63183 71.64182,-22.71807 10.80219,-2.84372 17.88236,-11.073 21.27228,-10.05511 12.27112,-0.89102 13.76383,25.66912 5.86784,27.62595 -31.09527,4.56377 -48.76316,2.63772 -79.85842,11.44328 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
        <path
           sodipodi:nodetypes="ccsccccsc"
           inkscape:connector-curvature="0"
           id="path5488"
           d="m 165.44117,679.72349 c 0.55992,-10.21772 -8.38843,-6.91831 -11.02734,3.18557 3.16535,28.79842 -18.621,73.62164 5.90075,102.0204 11.2851,13.06933 72.50888,13.30221 84.39146,4.71358 17.01662,-14.57154 7.85274,-28.29435 -2.36625,-27.88037 -19.38336,1.4505 -19.42048,-50.0732 -27.57994,-48.89888 -8.69404,1.63399 0.64114,48.83811 -6.38105,54.13434 -11.27931,8.35714 -30.81168,-3.74639 -36.18692,-11.11154 -14.7811,-20.25304 -3.422,-56.24978 -6.75071,-76.1631 z"
           style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      </g>
    </g>
  </g>
</svg>
`,S:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Nan.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Nan.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="10.088872"
     inkscape:cy="351.69723"
     inkscape:document-units="px"
     inkscape:current-layer="g4184"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="-8"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="85.681178,200.03932"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4184"
       transform="matrix(1.0044125,0,0,1.0044125,-10.100723,-17.440744)">
      <g
         id="g4195"
         transform="matrix(0.96971821,0,0,1,4.8296528,0)">
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 180.01019,689.47005 c -6.60859,-0.48504 -30.38318,25.26489 -36.60586,34.32229 -5.69504,6.90501 4.79486,11.13002 8.86231,16.9841 5.13346,8.06071 1.22147,89.90666 -8.33752,135.13699 l 15.46785,-2.31045 c 10.15168,-37.07022 17.66151,-105.92002 25.26199,-112.07982 7.23438,-5.64883 23.67201,-5.05439 21.67531,-13.41661 -4.13317,-18.67217 -20.44655,-58.42532 -26.32408,-58.6365 z"
           id="path4166"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 122.01681,796.92731 c -18.91603,0.87973 -28.798248,10.69738 -1.04916,14.50033 32.69916,1.34438 100.914,-5.95887 100.99124,-15.21074 0.50007,-12.38604 -9.68021,-30.36115 -15.72706,-30.24973 -4.80999,0.0666 -14.60707,6.08199 -15.49735,13.35218 -0.7723,8.52441 -50.99279,16.5893 -68.71767,17.60796 z"
           id="path4168"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 153.67205,848.84527 c 25.63242,-10.5633 45.92829,-20.43381 70.16669,-33.78515 8.50421,-4.39856 16.45949,4.65195 14.71585,10.49483 -1.13333,3.79775 -14.08578,26.76789 -2.69599,27.66397 -10.29756,8.02971 -38.07021,31.79678 -54.00467,67.80402 l -24.83006,-2.09832 c 10.65568,-26.66627 51.36933,-71.18083 48.61096,-74.10112 -4.70145,-5.08631 -42.1547,13.5402 -50.3375,11.1914 z"
           id="path4170"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="ccscccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="M 58.289846,893.36579 C 98.37429,871.73604 187.21397,852.18031 258.85433,851.06873 c 9.11568,-0.5564 30.91343,52.37414 30.72079,61.51831 -0.59867,26.02805 -45.433,88.49816 -73.24718,126.11726 -5.79302,6.4227 -18.89804,4.8464 -27.69634,-0.6182 l -47.53702,-31.5645 c -4.80435,-4.0497 -3.00434,-12.59097 5.05004,-7.74355 10.64061,6.55005 32.405,14.85845 44.46529,16.43385 11.0746,1.4727 48.13752,-86.15338 50.35477,-100.92899 1.30281,-6.70884 -7.4831,-39.24423 -23.33301,-40.27292 -107.24408,3.99322 -143.146386,35.79646 -157.540505,37.24847 z"
           id="path4172"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccccccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 35.176769,856.0746 c -8.329143,2.47339 -5.543427,12.92357 -4.359061,19.23455 9.767401,43.24858 46.332868,125.35065 53.641585,130.24035 22.498047,12.4821 31.295027,4.1632 30.912857,-5.2458 -1.13661,-9.1203 -30.281081,-84.44338 -48.575803,-124.19996 -6.317716,-12.70167 -22.0543,-22.43785 -31.619578,-20.02914 z"
           id="path4174"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 69.201953,835.83331 c -13.711354,-6.44835 -9.284101,-18.12879 9.262409,-11.09585 42.105238,15.7158 58.236628,41.97104 71.995068,93.68228 l -15.46785,4.19662 C 121.71411,873.69471 105.90056,855.54714 69.201953,835.83331 Z"
           id="path4176"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="ccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 121.2172,943.33039 c -15.90514,5.90334 -13.03153,-18.01065 -2.259,-21.03137 74.13079,-23.14959 77.60464,-21.1582 89.99832,-16.23043 11.85218,4.7626 16.09417,21.95025 0.35075,21.09865 -31.18654,-1.96893 -55.58683,2.54929 -88.09007,16.16315 z"
           id="path4178"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="ccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 124.2573,959.25452 c -10.22238,2.70755 -7.0172,18.26947 3.27761,15.95048 26.7444,-5.66326 45.57025,-9.03231 73.05055,-12.99668 8.28191,-1.37467 8.02256,-22.57795 0.5386,-25.88353 -10.0892,-3.02726 -11.21854,3.61802 -15.65202,6.11282 -12.14009,6.20765 -37.21982,10.4483 -61.21474,16.81691 z"
           id="path4180"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cccccc" />
        <path
           style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
           d="m 146.80164,925.40788 c 2.23284,25.80046 4.91455,54.74838 1.7486,80.54882 l 17.52265,9.6443 c 3.12723,-29.67464 6.5145,-63.19621 4.2514,-92.87087 z"
           id="path4182"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="ccccc" />
      </g>
    </g>
  </g>
</svg>
`,N:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pei.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pei.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="2.0219278"
     inkscape:cx="195.95216"
     inkscape:cy="272.83134"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g5727"
       transform="matrix(0.93642236,0,0,0.93642236,5.154253,91.360642)">
      <path
         sodipodi:nodetypes="cccc"
         inkscape:connector-curvature="0"
         id="path5703"
         d="m 29.907667,770.38167 c -6.194152,8.02265 21.428513,29.44188 41.676198,35.8917 6.543131,2.43905 26.830033,-18.71391 18.905069,-23.55726 C 68.697873,771.4298 34.310316,764.43141 29.907667,770.38167 Z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccsscccccscc"
         inkscape:connector-curvature="0"
         id="path5705"
         d="M 67.104158,774.19112 C 42.13506,868.07692 49.447167,858.11075 6.4295069,915.85877 -0.50548575,933.44331 22.298955,948.55984 33.463664,941.83903 48.915156,931.49238 66.363769,919.56071 77.251814,906.15771 97.02135,881.82175 187.80142,814.17483 249.20824,795.20483 c 12.36647,-3.82029 23.89371,-3.27102 35.1811,-5.71061 8.48368,-2.41164 -0.039,-15.01848 -4.69849,-19.79037 -15.70503,-16.54927 -43.36555,-26.34031 -52.42522,-19.8166 -3.97499,1.96567 1.92993,29.4811 -2.89142,32.06441 -46.59775,22.80796 -132.985525,81.67515 -172.248124,111.81826 -1.469515,0.68531 1.342551,-5.30234 2.312223,-7.24359 8.626121,-17.26919 18.010728,-65.73019 22.724979,-115.03932 0.446147,-11.0793 -8.412178,-7.22542 -10.05913,2.70411 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cscscccsccc"
         inkscape:connector-curvature="0"
         id="path5707"
         d="m 114.09719,711.59755 c -10.01917,-1.06336 -15.49032,8.80873 -14.342756,18.46301 0.973226,8.1876 1.251676,18.6738 9.832116,27.56108 5.64676,6.99777 1.45164,60.0977 1.34299,105.00662 -0.008,3.14771 -2.4216,3.81435 -2.3676,6.84866 -0.0102,7.91188 4.13472,27.85644 9.27843,28.02083 4.80428,-0.23729 11.74209,-7.6051 12.10232,-18.15965 2.70642,-49.99586 7.29702,-115.74474 11.25622,-124.14954 4.98852,-10.58991 13.58939,-7.74231 14.45015,-17.85614 1.2724,-14.15477 -19.13105,-22.60255 -41.55189,-25.73487 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccccccc"
         inkscape:connector-curvature="0"
         id="path5725"
         d="m 185.33231,661.72592 c -11.04102,2e-5 -32.46283,30.3394 -32.52388,36.72052 0.11709,8.1973 20.7411,15.69941 20.32596,27.9071 -0.39117,69.66027 -18.68319,181.14057 -4.68375,199.36122 20.54014,24.53708 118.85312,1.86586 124.94486,-3.14748 9.69372,-7.94512 13.91122,-15.91222 10.49157,-23.95576 -3.66499,-8.85428 -76.70142,11.15833 -108.85778,-0.69943 -13.49873,-4.95729 -3.98491,-100.61173 -3.75176,-156.79993 0.6848,-12.78519 25.76648,-27.26784 25.87921,-32.87359 0.39967,-10.88258 -18.36945,-45.20268 -31.82443,-46.51265 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"1p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin1.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin1.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="97.181841"
     inkscape:cy="219.79673"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g6641"
       transform="matrix(0.89332354,0,0,0.89332354,-207.38654,341.03071)">
      <circle
         r="160"
         cy="572.36218"
         cx="400"
         id="path5446"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#000000;stroke-width:10.03299999;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="145"
         cy="572.36218"
         cx="400"
         id="path5448"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:#000037;stroke-width:10.03299999;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <g
         transform="translate(0.35355339,0)"
         id="g5971">
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.95532227;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 392.32741,496.48087 c -8.27246,-82.93549 24.08267,-82.93853 15.28731,0"
           id="path5460"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cc"
           inkscape:transform-center-y="-122.30186"
           inkscape:tile-cx="399.93275"
           inkscape:tile-cy="572.50692"
           inkscape:tile-w="30.803226"
           inkscape:tile-h="64.041819"
           inkscape:tile-x0="384.53113"
           inkscape:tile-y0="430.98601" />
        <g
           id="g5914"
           transform="translate(-31,-64.5)">
          <use
            
            
             inkscape:transform-center-y="-100.94097"
             inkscape:transform-center-x="-33.466979"
             id="use5880"
             transform="matrix(0.93969262,0.34202014,-0.34202014,0.93969262,250.39181,-36.968234)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-80.402085"
             inkscape:transform-center-x="-66.418446"
             id="use5882"
             transform="matrix(0.76604444,0.64278761,-0.64278761,0.76604444,491.62428,-58.031527)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-50.569105"
             inkscape:transform-center-x="-92.035968"
             id="use5884"
             transform="matrix(0.5,0.8660254,-0.8660254,0.5,725.51269,4.6818112)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-16.020278"
             inkscape:transform-center-x="-106.83186"
             id="use5886"
             transform="matrix(0.17364818,0.98480775,-0.98480775,0.17364818,923.8467,143.6076)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="16.01696"
             inkscape:transform-center-x="-106.83192"
             id="use5888"
             transform="matrix(-0.17364818,0.98480775,-0.98480775,-0.17364818,1062.7042,341.98942)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="50.568001"
             inkscape:transform-center-x="-92.03626"
             id="use5890"
             transform="matrix(-0.5,0.8660254,-0.8660254,-0.5,1125.3371,575.89941)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="80.401655"
             inkscape:transform-center-x="-66.419137"
             id="use5892"
             transform="matrix(-0.76604444,0.64278761,-0.64278761,-0.76604444,1104.1907,817.1246)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="100.94079"
             inkscape:transform-center-x="-33.468884"
             id="use5894"
             transform="matrix(-0.93969262,0.34202014,-0.34202014,-0.93969262,1001.8161,1036.5696)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="109.52358"
             id="use5896"
             transform="matrix(-1,0,0,-1,830.56043,1207.7663)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="100.94095"
             inkscape:transform-center-x="33.466976"
             id="use5898"
             transform="matrix(-0.93969262,-0.34202014,0.34202014,-0.93969262,611.08014,1310.0656)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="80.402089"
             inkscape:transform-center-x="66.418443"
             id="use5900"
             transform="matrix(-0.76604444,-0.64278761,0.64278761,-0.76604444,369.84767,1331.1292)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="50.569089"
             inkscape:transform-center-x="92.035965"
             id="use5902"
             transform="matrix(-0.5,-0.8660254,0.8660254,-0.5,135.95926,1268.4156)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="16.020282"
             inkscape:transform-center-x="106.83186"
             id="use5904"
             transform="matrix(-0.17364818,-0.98480775,0.98480775,-0.17364818,-62.374757,1129.4899)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-16.016956"
             inkscape:transform-center-x="106.83194"
             id="use5906"
             transform="matrix(0.17364818,-0.98480775,0.98480775,0.17364818,-201.23234,931.10806)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-50.567997"
             inkscape:transform-center-x="92.036247"
             id="use5908"
             transform="matrix(0.5,-0.8660254,0.8660254,0.5,-263.86519,697.19807)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-80.401651"
             inkscape:transform-center-x="66.419125"
             id="use5910"
             transform="matrix(0.76604444,-0.64278761,0.64278761,0.76604444,-242.71892,455.97288)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
          <use
            
            
             inkscape:transform-center-y="-100.94079"
             inkscape:transform-center-x="33.468911"
             id="use5912"
             transform="matrix(0.93969262,-0.34202014,0.34202014,0.93969262,-140.34405,236.5278)"
             xlink:href="#path5460"
             inkscape:tiled-clone-of="#path5460"
             y="0"
             x="0" />
        </g>
      </g>
      <circle
         r="84.473503"
         cy="572.36218"
         cx="400"
         id="path5454"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#000037;stroke-width:10.23999977;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="72.314705"
         cy="572.36218"
         cx="400"
         id="path5992"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <g
         inkscape:transform-center-y="-33.991657"
         inkscape:transform-center-x="-0.057936225"
         transform="translate(-0.17677669,0.08838835)"
         id="g6038">
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:23.10970306;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 401.33152,554.24492 c -3.14542,-9.94018 -29.56798,-32.56818 -1.22213,-34.01501 29.36983,1.60064 0.87135,23.52895 -1.71866,34.17839"
           id="path5994"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="ccc"
           inkscape:transform-center-x="-0.0026845101"
           inkscape:transform-center-y="63.28125" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:6;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 400,555.19145 0,-25.65771"
           id="path6018"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cc" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path6028"
           cx="400"
           cy="527.83356"
           r="4.818634" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:4;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 404.6875,557.4872 6.5651,-16.10951"
           id="path6030"
           inkscape:connector-curvature="0"
           sodipodi:nodetypes="cc" />
        <use
           x="0"
           y="0"
           xlink:href="#path6030"
           id="use6032"
           inkscape:transform-center-x="7.5"
          
          
           transform="matrix(-1,0,0,1,799.9375,0)" />
      </g>
      <circle
         r="16.034555"
         cy="572.36218"
         cx="400"
         id="path6010"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#b93c3c;stroke-width:6.66957808;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="8.5365553"
         cy="572.36218"
         cx="400"
         id="path6016"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:6;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <use
        
        
         transform="matrix(0.30901699,-0.95105652,0.95105652,0.30901699,-267.97075,775.88681)"
         id="use6047"
         inkscape:transform-center-y="-12.637547"
         inkscape:transform-center-x="31.767912"
         xlink:href="#g6038"
         y="0"
         x="0" />
      <use
        
        
         transform="matrix(0.30901699,-0.95105652,0.95105652,0.30901699,-267.97075,775.88681)"
         id="use6049"
         inkscape:transform-center-y="27.707643"
         inkscape:transform-center-x="21.376967"
         xlink:href="#use6047"
         y="0"
         x="0" />
      <use
        
        
         transform="matrix(0.30901699,-0.95105652,0.95105652,0.30901699,-267.97075,775.88681)"
         id="use6051"
         inkscape:transform-center-y="27.656408"
         inkscape:transform-center-x="-21.44051"
         xlink:href="#use6049"
         y="0"
         x="0" />
      <use
        
        
         transform="matrix(0.30901699,-0.95105652,0.95105652,0.30901699,-267.97075,775.88681)"
         id="use6053"
         inkscape:transform-center-y="-12.723383"
         inkscape:transform-center-x="-31.665728"
         xlink:href="#use6051"
         y="0"
         x="0" />
    </g>
  </g>
</svg>
`,"2p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin2.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin2.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="243.05578"
     inkscape:cy="172.62628"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7625"
       transform="translate(-3.570741e-7,-0.81785177)">
      <g
         transform="matrix(0.76584228,0,0,0.76584228,512.08203,490.47575)"
         id="g4360">
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#000000;stroke-width:17.21711159;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path15137"
           cx="-472.78326"
           cy="341.25565"
           r="107.56709" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:0;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:0"
           id="path15137-2"
           cx="-472.78326"
           cy="341.62653"
           r="87.813393" />
        <g
           id="g4247"
           transform="matrix(0.10477442,0,0,0.08587532,-506.32882,359.1076)"
           inkscape:transform-center-x="-8.4743063">
          <path
             d="m -93.329197,-74.125463 a 163.9187,163.9187 0 0 1 -224.595053,-13.081173 163.9187,163.9187 0 0 1 0,-224.975674 163.9187,163.9187 0 0 1 224.595055,-13.08117"
             sodipodi:open="true"
             sodipodi:end="5.4105207"
             sodipodi:start="0.87266463"
             sodipodi:ry="163.9187"
             sodipodi:rx="163.9187"
             sodipodi:cy="-199.69447"
             sodipodi:cx="-198.69411"
             sodipodi:type="arc"
             mask="none"
             style="opacity:1;fill:none;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:125.71019745;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
             id="circle15184" />
        </g>
        <use
           x="0"
           y="0"
           xlink:href="#g4247"
           inkscape:transform-center-x="27.647557"
           id="use4252"
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-532.66673,-238.47425)"
          
          
           inkscape:transform-center-y="57.680228" />
        <use
           x="0"
           y="0"
           xlink:href="#use4252"
           inkscape:transform-center-x="-28.777383"
           inkscape:transform-center-y="57.02792"
           id="use4254"
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
          
           />
        <use
           x="0"
           y="0"
           xlink:href="#use4254"
           inkscape:transform-center-x="-70.647623"
           inkscape:transform-center-y="-0.74509288"
           id="use4256"
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
          
           />
        <use
           x="0"
           y="0"
           xlink:href="#use4256"
           inkscape:transform-center-x="-27.647551"
           inkscape:transform-center-y="-57.680226"
           id="use4258"
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
          
           />
        <use
           x="0"
           y="0"
           xlink:href="#use4258"
           inkscape:transform-center-x="28.777389"
           inkscape:transform-center-y="-57.027919"
           id="use4260"
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
          
           />
        <circle
           style="opacity:1;fill:none;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:12.28584385;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path15154-5"
           cx="-472.93054"
           cy="341.51578"
           r="18.188499" />
      </g>
      <g
         id="g7557"
         transform="matrix(0.76584228,0,0,0.76584228,512.08203,692.97575)">
        <circle
           r="107.56709"
           cy="341.25565"
           cx="-472.78326"
           id="circle7559"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:#000000;stroke-width:17.21711159;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="87.813393"
           cy="341.62653"
           cx="-472.78326"
           id="circle7561"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:0;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:0" />
        <g
           inkscape:transform-center-x="-8.4743063"
           transform="matrix(0.10477442,0,0,0.08587532,-506.32882,359.1076)"
           id="g7563">
          <path
             d="m -93.329197,-74.125463 a 163.9187,163.9187 0 0 1 -224.595053,-13.081173 163.9187,163.9187 0 0 1 0,-224.975674 163.9187,163.9187 0 0 1 224.595055,-13.08117"
             id="path7565"
             style="opacity:1;fill:none;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:125.71019745;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
             mask="none"
             sodipodi:type="arc"
             sodipodi:cx="-198.69411"
             sodipodi:cy="-199.69447"
             sodipodi:rx="163.9187"
             sodipodi:ry="163.9187"
             sodipodi:start="0.87266463"
             sodipodi:end="5.4105207"
             sodipodi:open="true" />
        </g>
        <use
           inkscape:transform-center-y="57.680228"
          
          
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-532.66673,-238.47425)"
           id="use7567"
           inkscape:transform-center-x="27.647557"
           xlink:href="#g4247"
           y="0"
           x="0" />
        <use
          
          
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
           id="use7569"
           inkscape:transform-center-y="57.02792"
           inkscape:transform-center-x="-28.777383"
           xlink:href="#use4252"
           y="0"
           x="0" />
        <use
          
          
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
           id="use7571"
           inkscape:transform-center-y="-0.74509288"
           inkscape:transform-center-x="-70.647623"
           xlink:href="#use4254"
           y="0"
           x="0" />
        <use
          
          
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
           id="use7573"
           inkscape:transform-center-y="-57.680226"
           inkscape:transform-center-x="-27.647551"
           xlink:href="#use4256"
           y="0"
           x="0" />
        <use
          
          
           transform="matrix(0.5,-0.8660254,0.8660254,0.5,-533.74838,-239.03721)"
           id="use7575"
           inkscape:transform-center-y="-57.027919"
           inkscape:transform-center-x="28.777389"
           xlink:href="#use4258"
           y="0"
           x="0" />
        <circle
           r="18.188499"
           cy="341.51578"
           cx="-472.93054"
           id="circle7577"
           style="opacity:1;fill:none;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:12.28584385;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
    </g>
  </g>
</svg>
`,"3p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin3.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin3.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="66.856772"
     inkscape:cy="130.39583"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7705"
       transform="translate(4.6993944e-7,0.02106803)">
      <g
         id="g7649"
         transform="matrix(1.0227353,0,0,1.0227353,286.16819,231.36176)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7651"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7653"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7655"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7657"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7659"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7661"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,360.65067,355.76781)"
         id="g7677">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7679"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7681"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7683"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7685"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7687"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7689"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         inkscape:transform-center-y="124.46875"
         inkscape:transform-center-x="-74.484372"
         transform="matrix(1.0227353,0,0,1.0227353,435.13694,480.29928)"
         id="g7691">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7693"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7695"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7697"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7699"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7701"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7703"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
    </g>
  </g>
</svg>
`,"4p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin4.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin4.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="75.439196"
     inkscape:cy="244.23776"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7912"
       transform="translate(4.6993944e-7,-1535.3225)">
      <g
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,1781.0186)"
         id="g7840">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7842"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7844"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7846"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7848"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7850"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7852"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         id="g7854"
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,1781.0186)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7856"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7858"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7860"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7862"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7864"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7866"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         id="g7868"
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,2000.9496)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7870"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7872"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7874"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7876"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7879"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7881"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,2000.9496)"
         id="g7883">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7885"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7887"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7889"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7891"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7893"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7895"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
    </g>
  </g>
</svg>
`,"5pr":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin5-Dora.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Pin5-Dora.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.71485945"
     inkscape:cx="-162.1301"
     inkscape:cy="301.89564"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7942"
       transform="translate(4.6993944e-7,0.67748)">
      <g
         id="g7728"
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,245.01862)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7730"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7732"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7734"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7736"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7738"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7740"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,245.01862)"
         id="g7784">
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7786"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7788"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7790"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7792"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7794"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7796"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,464.94965)"
         id="g7798">
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7800"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7802"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7804"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7806"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7808"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7810"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         id="g7812"
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,464.94965)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7814"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7816"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7818"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7820"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7822"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7824"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,360.63962,355.83523)"
         id="g7826">
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7828"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7830"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7832"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7834"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7836"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7838"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
    </g>
    <circle
       style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="path4178"
       cx="46.162918"
       cy="852.35797"
       r="19.7831" />
  </g>
</svg>
`,"5p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin5.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin5.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="330.46142"
     inkscape:cy="158.01828"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7942"
       transform="translate(4.6993944e-7,0.67748)">
      <g
         id="g7728"
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,245.01862)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7730"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7732"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7734"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7736"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7738"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7740"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,245.01862)"
         id="g7784">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7786"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7788"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7790"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7792"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7794"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7796"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,464.94965)"
         id="g7798">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7800"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7802"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7804"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7806"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7808"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7810"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         id="g7812"
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,464.94965)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7814"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7816"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7818"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7820"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7822"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7824"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,360.63962,355.83523)"
         id="g7826">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7828"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7830"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7832"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7834"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7836"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7838"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
    </g>
  </g>
</svg>
`,"6p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin6.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin6.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="182.96788"
     inkscape:cy="234.29201"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g8535"
       transform="translate(4.9975586e-7,-10.378157)">
      <circle
         r="61.117695"
         cy="736.81885"
         cx="100.45483"
         id="circle8511"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8100"
         cx="100.56846"
         cy="988.50354"
         r="61.117695" />
      <circle
         r="61.117695"
         cy="988.50354"
         cx="199.49802"
         id="circle8114"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8070"
         cx="199.49802"
         cy="901.17584"
         r="61.117695" />
      <circle
         r="61.117695"
         cy="901.17584"
         cx="100.56846"
         id="circle7981"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="46.094604"
         cy="901.17584"
         cx="100.56846"
         id="circle7983"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="901.17584"
         cx="100.56846"
         id="circle7985"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path7988"
         sodipodi:type="arc"
         sodipodi:cx="94.357384"
         sodipodi:cy="-901.84741"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 115.9051,-901.84741 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.937586,-20.73383 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701553,19.89445"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="901.17584"
         cx="100.56846"
         id="circle7990"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="901.17584"
         cx="100.56846"
         id="circle7992"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8072"
         cx="199.49802"
         cy="901.17584"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8074"
         cx="199.49802"
         cy="901.17584"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 214.83229,-902.52917 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.93758,-20.73383 21.547712,21.547714 0 0 1 20.31773,-22.32352 21.547712,21.547714 0 0 1 22.70155,19.89445"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-902.52917"
         sodipodi:cx="193.28458"
         sodipodi:type="arc"
         id="path8076"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8078"
         cx="199.49802"
         cy="901.17584"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8080"
         cx="199.49802"
         cy="901.17584"
         r="3.6676245" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8102"
         cx="100.56846"
         cy="988.50354"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8104"
         cx="100.56846"
         cy="988.50354"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 115.3033,-989.17303 a 21.547712,21.547714 0 0 1 -21.142609,21.5439 21.547712,21.547714 0 0 1 -21.937586,-20.73383 21.547712,21.547714 0 0 1 20.317733,-22.32354 21.547712,21.547714 0 0 1 22.701552,19.89447"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-989.17303"
         sodipodi:cx="93.755585"
         sodipodi:type="arc"
         id="path8106"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8108"
         cx="100.56846"
         cy="988.50354"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8110"
         cx="100.56846"
         cy="988.50354"
         r="3.6676245" />
      <circle
         r="46.094604"
         cy="988.50354"
         cx="199.49802"
         id="circle8116"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="988.50354"
         cx="199.49802"
         id="circle8118"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8120"
         sodipodi:type="arc"
         sodipodi:cx="192.68277"
         sodipodi:cy="-989.8548"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 214.23048,-989.8548 a 21.547712,21.547714 0 0 1 -21.1426,21.54391 21.547712,21.547714 0 0 1 -21.93759,-20.73384 21.547712,21.547714 0 0 1 20.31773,-22.32347 21.547712,21.547714 0 0 1 22.70155,19.89441"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="988.50354"
         cx="199.49802"
         id="circle8122"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="988.50354"
         cx="199.49802"
         id="circle8124"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8497"
         cx="199.62656"
         cy="736.81885"
         r="61.117695" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8499"
         cx="199.62656"
         cy="736.81885"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8501"
         cx="199.62656"
         cy="736.81885"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 216.09347,-738.177 a 21.547712,21.547714 0 0 1 -21.1426,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.73383 21.547712,21.547714 0 0 1 20.31773,-22.32352 21.547712,21.547714 0 0 1 22.70156,19.89445"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-738.177"
         sodipodi:cx="194.54576"
         sodipodi:type="arc"
         id="path8503"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8505"
         cx="199.62656"
         cy="736.81885"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8507"
         cx="199.62656"
         cy="736.81885"
         r="3.6676245" />
      <circle
         r="46.094604"
         cy="736.81885"
         cx="100.45483"
         id="circle8513"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="736.81885"
         cx="100.45483"
         id="circle8515"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8517"
         sodipodi:type="arc"
         sodipodi:cx="95.376396"
         sodipodi:cy="-737.49353"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 116.92411,-737.49353 a 21.547712,21.547714 0 0 1 -21.142607,21.54391 21.547712,21.547714 0 0 1 -21.937587,-20.73384 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701551,19.89446"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="736.81885"
         cx="100.45483"
         id="circle8519"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="736.81885"
         cx="100.45483"
         id="circle8521"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"7p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin7.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin7.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="277.61694"
     inkscape:cy="154.80916"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g8444"
       transform="translate(1.1865232e-8,-3009.7745)">
      <circle
         r="61.117695"
         cy="3997.95"
         cx="100.56846"
         id="circle8248"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8250"
         cx="199.49802"
         cy="3997.95"
         r="61.117695" />
      <circle
         r="61.117695"
         cy="3910.6223"
         cx="199.49802"
         id="circle8252"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8254"
         cx="232.19324"
         cy="3828.0625"
         r="61.117695" />
      <g
         id="g8256"
         transform="matrix(1.0227353,0,0,1.0227353,360.68434,3280.58)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle8258"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle8260"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle8262"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path8264"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle8266"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle8268"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <circle
         r="61.117695"
         cy="3777.1067"
         cx="150.02574"
         id="circle8270"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8272"
         cx="100.56846"
         cy="3910.6223"
         r="61.117695" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8274"
         cx="100.56846"
         cy="3910.6223"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8276"
         cx="100.56846"
         cy="3910.6223"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 95.166189,-3911.2224 a 21.547712,21.547714 0 0 1 -21.142606,21.5439 21.547712,21.547714 0 0 1 -21.937586,-20.7338 21.547712,21.547714 0 0 1 20.317733,-22.3236 21.547712,21.547714 0 0 1 22.701551,19.8945"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-3911.2224"
         sodipodi:cx="73.618477"
         sodipodi:type="arc"
         id="path8278"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8280"
         cx="100.56846"
         cy="3910.6223"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8282"
         cx="100.56846"
         cy="3910.6223"
         r="3.6676245" />
      <circle
         r="61.117695"
         cy="3726.1067"
         cx="67.790916"
         id="circle8284"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="46.094604"
         cy="3726.1067"
         cx="67.790916"
         id="circle8286"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="3726.1067"
         cx="67.790916"
         id="circle8288"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8290"
         sodipodi:type="arc"
         sodipodi:cx="42.113262"
         sodipodi:cy="-3726.4854"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 63.660975,-3726.4854 a 21.547712,21.547714 0 0 1 -21.142606,21.544 21.547712,21.547714 0 0 1 -21.937587,-20.7339 21.547712,21.547714 0 0 1 20.317733,-22.3235 21.547712,21.547714 0 0 1 22.701552,19.8945"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="3726.1067"
         cx="67.790916"
         id="circle8292"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="3726.1067"
         cx="67.790916"
         id="circle8294"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="46.094604"
         cy="3777.1067"
         cx="150.02574"
         id="circle8296"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="3777.1067"
         cx="150.02574"
         id="circle8298"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8300"
         sodipodi:type="arc"
         sodipodi:cx="123.99467"
         sodipodi:cy="-3778.0508"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 145.54238,-3778.0508 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.93758,-20.7338 21.547712,21.547714 0 0 1 20.31773,-22.3235 21.547712,21.547714 0 0 1 22.70155,19.8944"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="3777.1067"
         cx="150.02574"
         id="circle8302"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="3777.1067"
         cx="150.02574"
         id="circle8304"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8306"
         cx="232.19324"
         cy="3828.0625"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8308"
         cx="232.19324"
         cy="3828.0625"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 227.35679,-3829.5718 a 21.547712,21.547714 0 0 1 -21.1426,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.7338 21.547712,21.547714 0 0 1 20.31774,-22.3235 21.547712,21.547714 0 0 1 22.70155,19.8944"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-3829.5718"
         sodipodi:cx="205.80908"
         sodipodi:type="arc"
         id="path8310"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8312"
         cx="232.19324"
         cy="3828.0625"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8314"
         cx="232.19324"
         cy="3828.0625"
         r="3.6676245" />
      <circle
         r="46.094604"
         cy="3910.6223"
         cx="199.49802"
         id="circle8316"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="3910.6223"
         cx="199.49802"
         id="circle8318"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8320"
         sodipodi:type="arc"
         sodipodi:cx="172.54567"
         sodipodi:cy="-3911.9041"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 194.09338,-3911.9041 a 21.547712,21.547714 0 0 1 -21.1426,21.544 21.547712,21.547714 0 0 1 -21.93759,-20.7339 21.547712,21.547714 0 0 1 20.31773,-22.3235 21.547712,21.547714 0 0 1 22.70155,19.8945"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="3910.6223"
         cx="199.49802"
         id="circle8322"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="3910.6223"
         cx="199.49802"
         id="circle8324"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="46.094604"
         cy="3997.95"
         cx="100.56846"
         id="circle8326"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="3997.95"
         cx="100.56846"
         id="circle8328"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8330"
         sodipodi:type="arc"
         sodipodi:cx="73.016678"
         sodipodi:cy="-3998.5481"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 94.56439,-3998.5481 a 21.547712,21.547714 0 0 1 -21.142606,21.5439 21.547712,21.547714 0 0 1 -21.937586,-20.7338 21.547712,21.547714 0 0 1 20.317733,-22.3235 21.547712,21.547714 0 0 1 22.701551,19.8944"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="3997.95"
         cx="100.56846"
         id="circle8332"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="3997.95"
         cx="100.56846"
         id="circle8334"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8336"
         cx="199.49802"
         cy="3997.95"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8338"
         cx="199.49802"
         cy="3997.95"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 193.49158,-3999.2297 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.7339 21.547712,21.547714 0 0 1 20.31774,-22.3235 21.547712,21.547714 0 0 1 22.70155,19.8945"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-3999.2297"
         sodipodi:cx="171.94386"
         sodipodi:type="arc"
         id="path8340"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8342"
         cx="199.49802"
         cy="3997.95"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8344"
         cx="199.49802"
         cy="3997.95"
         r="3.6676245" />
    </g>
  </g>
</svg>
`,"8p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin8.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin8.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-9" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="175.29364"
     inkscape:cy="240.90751"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g8749"
       transform="translate(-2.2924804e-7,0.0010341)">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8575"
         cx="97.785736"
         cy="718.12097"
         r="61.117695" />
      <circle
         r="61.117695"
         cy="808.12097"
         cx="97.785736"
         id="circle8589"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="61.117695"
         cy="896.62097"
         cx="97.785736"
         id="circle8619"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8633"
         cx="97.785736"
         cy="986.62097"
         r="61.117695" />
      <circle
         r="61.117695"
         cy="718.12097"
         cx="202.21381"
         id="circle8647"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8661"
         cx="202.21381"
         cy="808.12097"
         r="61.117695" />
      <circle
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8675"
         cx="202.21381"
         cy="896.62097"
         r="61.117695" />
      <circle
         r="61.117695"
         cy="986.62097"
         cx="202.21381"
         id="circle8689"
         style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8577"
         cx="97.785736"
         cy="718.12097"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8579"
         cx="97.785736"
         cy="718.12097"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 114.38392,-718.77771 a 21.547712,21.547714 0 0 1 -21.142601,21.54391 21.547712,21.547714 0 0 1 -21.937587,-20.73384 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701555,19.89446"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-718.77771"
         sodipodi:cx="92.836212"
         sodipodi:type="arc"
         id="path8581"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8583"
         cx="97.785736"
         cy="718.12097"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8585"
         cx="97.785736"
         cy="718.12097"
         r="3.6676245" />
      <circle
         r="46.094604"
         cy="808.12097"
         cx="97.785736"
         id="circle8591"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="808.12097"
         cx="97.785736"
         id="circle8593"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8595"
         sodipodi:type="arc"
         sodipodi:cx="92.216003"
         sodipodi:cy="-808.77557"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 113.76372,-808.77557 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.937587,-20.73383 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701554,19.89445"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="808.12097"
         cx="97.785736"
         id="circle8597"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="808.12097"
         cx="97.785736"
         id="circle8599"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="46.094604"
         cy="896.62097"
         cx="97.785736"
         id="circle8621"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="896.62097"
         cx="97.785736"
         id="circle8623"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8625"
         sodipodi:type="arc"
         sodipodi:cx="91.606125"
         sodipodi:cy="-897.27344"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 113.15384,-897.27344 a 21.547712,21.547714 0 0 1 -21.142609,21.54391 21.547712,21.547714 0 0 1 -21.937586,-20.73384 21.547712,21.547714 0 0 1 20.317733,-22.32351 21.547712,21.547714 0 0 1 22.701552,19.89445"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="896.62097"
         cx="97.785736"
         id="circle8627"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="896.62097"
         cx="97.785736"
         id="circle8629"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8635"
         cx="97.785736"
         cy="986.62097"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8637"
         cx="97.785736"
         cy="986.62097"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 112.53362,-987.2713 a 21.547712,21.547714 0 0 1 -21.142605,21.5439 21.547712,21.547714 0 0 1 -21.937586,-20.73383 21.547712,21.547714 0 0 1 20.317733,-22.32347 21.547712,21.547714 0 0 1 22.701548,19.8944"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-987.2713"
         sodipodi:cx="90.985909"
         sodipodi:type="arc"
         id="path8639"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8641"
         cx="97.785736"
         cy="986.62097"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8643"
         cx="97.785736"
         cy="986.62097"
         r="3.6676245" />
      <circle
         r="46.094604"
         cy="718.12097"
         cx="202.21381"
         id="circle8649"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="718.12097"
         cx="202.21381"
         id="circle8651"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8653"
         sodipodi:type="arc"
         sodipodi:cx="197.26181"
         sodipodi:cy="-719.49738"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 218.80952,-719.49738 a 21.547712,21.547714 0 0 1 -21.1426,21.54391 21.547712,21.547714 0 0 1 -21.93759,-20.73384 21.547712,21.547714 0 0 1 20.31773,-22.32351 21.547712,21.547714 0 0 1 22.70155,19.89445"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="718.12097"
         cx="202.21381"
         id="circle8655"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="718.12097"
         cx="202.21381"
         id="circle8657"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8663"
         cx="202.21381"
         cy="808.12097"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8665"
         cx="202.21381"
         cy="808.12097"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 218.1893,-809.49518 a 21.547712,21.547714 0 0 1 -21.14261,21.54391 21.547712,21.547714 0 0 1 -21.93758,-20.73384 21.547712,21.547714 0 0 1 20.31773,-22.32351 21.547712,21.547714 0 0 1 22.70155,19.89445"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-809.49518"
         sodipodi:cx="196.64159"
         sodipodi:type="arc"
         id="path8667"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8669"
         cx="202.21381"
         cy="808.12097"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8671"
         cx="202.21381"
         cy="808.12097"
         r="3.6676245" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8677"
         cx="202.21381"
         cy="896.62097"
         r="46.094604" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8679"
         cx="202.21381"
         cy="896.62097"
         r="39.692886" />
      <path
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
         d="m 217.57942,-897.99304 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.93758,-20.73383 21.547712,21.547714 0 0 1 20.31773,-22.32352 21.547712,21.547714 0 0 1 22.70155,19.89445"
         sodipodi:open="true"
         sodipodi:end="6.2079791"
         sodipodi:start="0"
         sodipodi:ry="21.547714"
         sodipodi:rx="21.547712"
         sodipodi:cy="-897.99304"
         sodipodi:cx="196.03171"
         sodipodi:type="arc"
         id="path8681"
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8683"
         cx="202.21381"
         cy="896.62097"
         r="8.0316467" />
      <circle
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle8685"
         cx="202.21381"
         cy="896.62097"
         r="3.6676245" />
      <circle
         r="46.094604"
         cy="986.62097"
         cx="202.21381"
         id="circle8691"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="39.692886"
         cy="986.62097"
         cx="202.21381"
         id="circle8693"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
         id="path8695"
         sodipodi:type="arc"
         sodipodi:cx="195.4115"
         sodipodi:cy="-987.99091"
         sodipodi:rx="21.547712"
         sodipodi:ry="21.547714"
         sodipodi:start="0"
         sodipodi:end="6.2079791"
         sodipodi:open="true"
         d="m 216.95921,-987.99091 a 21.547712,21.547714 0 0 1 -21.1426,21.54391 21.547712,21.547714 0 0 1 -21.93759,-20.73384 21.547712,21.547714 0 0 1 20.31773,-22.32356 21.547712,21.547714 0 0 1 22.70155,19.8945"
         transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
      <circle
         r="8.0316467"
         cy="986.62097"
         cx="202.21381"
         id="circle8697"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="3.6676245"
         cy="986.62097"
         cx="202.21381"
         id="circle8699"
         style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"9p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin9.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Pin9.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-9" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4142136"
     inkscape:cx="165.90933"
     inkscape:cy="215.8738"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7443"
       transform="translate(1.1328126e-8,0.37391082)">
      <g
         id="g7305">
        <circle
           r="61.117695"
           cy="714.98895"
           cx="237.42392"
           id="circle7275"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7261"
           cx="62.768543"
           cy="714.98895"
           r="61.117695" />
        <circle
           r="61.117695"
           cy="714.98895"
           cx="150.09624"
           id="circle7247"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="46.094604"
           cy="714.98895"
           cx="150.09624"
           id="circle7249"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="39.692886"
           cy="714.98895"
           cx="150.09624"
           id="circle7251"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
           id="path7253"
           sodipodi:type="arc"
           sodipodi:cx="145.16705"
           sodipodi:cy="-716.00623"
           sodipodi:rx="21.547712"
           sodipodi:ry="21.547714"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m 166.71477,-716.00623 a 21.547712,21.547714 0 0 1 -21.14261,21.54391 21.547712,21.547714 0 0 1 -21.93759,-20.73384 21.547712,21.547714 0 0 1 20.31774,-22.32351 21.547712,21.547714 0 0 1 22.70155,19.89445"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="8.0316467"
           cy="714.98895"
           cx="150.09624"
           id="circle7255"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.6676245"
           cy="714.98895"
           cx="150.09624"
           id="circle7257"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7263"
           cx="62.768543"
           cy="714.98895"
           r="46.094604" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7265"
           cx="62.768543"
           cy="714.98895"
           r="39.692886" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m 79.389147,-715.40442 a 21.547712,21.547714 0 0 1 -21.142606,21.54391 21.547712,21.547714 0 0 1 -21.937586,-20.73384 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701551,19.89446"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.547714"
           sodipodi:rx="21.547712"
           sodipodi:cy="-715.40442"
           sodipodi:cx="57.841434"
           sodipodi:type="arc"
           id="path7267"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7269"
           cx="62.768543"
           cy="714.98895"
           r="8.0316467" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7271"
           cx="62.768543"
           cy="714.98895"
           r="3.6676245" />
        <circle
           r="46.094604"
           cy="714.98895"
           cx="237.42392"
           id="circle7277"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="39.692886"
           cy="714.98895"
           cx="237.42392"
           id="circle7279"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
           id="path7281"
           sodipodi:type="arc"
           sodipodi:cx="232.49266"
           sodipodi:cy="-716.60803"
           sodipodi:rx="21.547712"
           sodipodi:ry="21.547714"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m 254.04037,-716.60803 a 21.547712,21.547714 0 0 1 -21.1426,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.73383 21.547712,21.547714 0 0 1 20.31773,-22.32352 21.547712,21.547714 0 0 1 22.70155,19.89445"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="8.0316467"
           cy="714.98895"
           cx="237.42392"
           id="circle7283"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.6676245"
           cy="714.98895"
           cx="237.42392"
           id="circle7285"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="translate(0,274)"
         id="g7325">
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7327"
           cx="237.42392"
           cy="714.98895"
           r="61.117695" />
        <circle
           r="61.117695"
           cy="714.98895"
           cx="62.768543"
           id="circle7329"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7331"
           cx="150.09624"
           cy="714.98895"
           r="61.117695" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7333"
           cx="150.09624"
           cy="714.98895"
           r="46.094604" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7335"
           cx="150.09624"
           cy="714.98895"
           r="39.692886" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m 166.71477,-716.00623 a 21.547712,21.547714 0 0 1 -21.14261,21.54391 21.547712,21.547714 0 0 1 -21.93759,-20.73384 21.547712,21.547714 0 0 1 20.31774,-22.32351 21.547712,21.547714 0 0 1 22.70155,19.89445"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.547714"
           sodipodi:rx="21.547712"
           sodipodi:cy="-716.00623"
           sodipodi:cx="145.16705"
           sodipodi:type="arc"
           id="path7337"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7339"
           cx="150.09624"
           cy="714.98895"
           r="8.0316467" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7341"
           cx="150.09624"
           cy="714.98895"
           r="3.6676245" />
        <circle
           r="46.094604"
           cy="714.98895"
           cx="62.768543"
           id="circle7343"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="39.692886"
           cy="714.98895"
           cx="62.768543"
           id="circle7345"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
           id="path7347"
           sodipodi:type="arc"
           sodipodi:cx="57.841434"
           sodipodi:cy="-715.40442"
           sodipodi:rx="21.547712"
           sodipodi:ry="21.547714"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m 79.389147,-715.40442 a 21.547712,21.547714 0 0 1 -21.142606,21.54391 21.547712,21.547714 0 0 1 -21.937586,-20.73384 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701551,19.89446"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="8.0316467"
           cy="714.98895"
           cx="62.768543"
           id="circle7349"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.6676245"
           cy="714.98895"
           cx="62.768543"
           id="circle7351"
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7353"
           cx="237.42392"
           cy="714.98895"
           r="46.094604" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7355"
           cx="237.42392"
           cy="714.98895"
           r="39.692886" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m 254.04037,-716.60803 a 21.547712,21.547714 0 0 1 -21.1426,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.73383 21.547712,21.547714 0 0 1 20.31773,-22.32352 21.547712,21.547714 0 0 1 22.70155,19.89445"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.547714"
           sodipodi:rx="21.547712"
           sodipodi:cy="-716.60803"
           sodipodi:cx="232.49266"
           sodipodi:type="arc"
           id="path7357"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7359"
           cx="237.42392"
           cy="714.98895"
           r="8.0316467" />
        <circle
           style="opacity:1;fill:#000037;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7361"
           cx="237.42392"
           cy="714.98895"
           r="3.6676245" />
      </g>
      <g
         id="g7423">
        <circle
           r="61.117695"
           cy="852.40082"
           cx="63.028664"
           id="circle7379"
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7393"
           cx="237.68404"
           cy="852.40082"
           r="61.117695" />
        <circle
           style="opacity:1;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7365"
           cx="150.00279"
           cy="852.40082"
           r="61.117695" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7367"
           cx="150.00279"
           cy="852.40082"
           r="46.094604" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7369"
           cx="150.00279"
           cy="852.40082"
           r="39.692886" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m 165.67438,-853.41418 a 21.547712,21.547714 0 0 1 -21.14261,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.73383 21.547712,21.547714 0 0 1 20.31774,-22.32352 21.547712,21.547714 0 0 1 22.70155,19.89445"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.547714"
           sodipodi:rx="21.547712"
           sodipodi:cy="-853.41418"
           sodipodi:cx="144.12666"
           sodipodi:type="arc"
           id="path7371"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7373"
           cx="150.00279"
           cy="852.40082"
           r="8.0316467" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7375"
           cx="150.00279"
           cy="852.40082"
           r="3.6676245" />
        <circle
           r="46.094604"
           cy="852.40082"
           cx="63.028664"
           id="circle7381"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="39.692886"
           cy="852.40082"
           cx="63.028664"
           id="circle7383"
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1"
           id="path7385"
           sodipodi:type="arc"
           sodipodi:cx="57.154606"
           sodipodi:cy="-852.81482"
           sodipodi:rx="21.547712"
           sodipodi:ry="21.547714"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m 78.702318,-852.81482 a 21.547712,21.547714 0 0 1 -21.142606,21.54391 21.547712,21.547714 0 0 1 -21.937586,-20.73384 21.547712,21.547714 0 0 1 20.317733,-22.32352 21.547712,21.547714 0 0 1 22.701551,19.89446"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="8.0316467"
           cy="852.40082"
           cx="63.028664"
           id="circle7387"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.6676245"
           cy="852.40082"
           cx="63.028664"
           id="circle7389"
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7395"
           cx="237.68404"
           cy="852.40082"
           r="46.094604" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7397"
           cx="237.68404"
           cy="852.40082"
           r="39.692886" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m 253.35354,-854.01843 a 21.547712,21.547714 0 0 1 -21.1426,21.5439 21.547712,21.547714 0 0 1 -21.93759,-20.73383 21.547712,21.547714 0 0 1 20.31773,-22.32352 21.547712,21.547714 0 0 1 22.70156,19.89445"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.547714"
           sodipodi:rx="21.547712"
           sodipodi:cy="-854.01843"
           sodipodi:cx="231.80583"
           sodipodi:type="arc"
           id="path7399"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.75044918;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.50089867, 13.50089867;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7401"
           cx="237.68404"
           cy="852.40082"
           r="8.0316467" />
        <circle
           style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7403"
           cx="237.68404"
           cy="852.40082"
           r="3.6676245" />
      </g>
    </g>
  </g>
</svg>
`,W:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Shaa.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Shaa.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="-92.357138"
     inkscape:cy="342.0988"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4192"
       transform="matrix(0.85008915,0,0,0.93538583,19.53046,57.10894)">
      <path
         sodipodi:nodetypes="cccccccc"
         inkscape:connector-curvature="0"
         id="path4166"
         d="m 18.30334,756.6 c -9.5861174,1.25345 -21.0440188,-11.12964 -7.344369,-14.65998 43.689934,-8.44524 191.763199,-31.91375 208.715039,-38.96975 14.88198,-4.47664 48.7794,19.64597 69.77151,30.82355 13.03521,7.62589 7.09752,16.27427 -2.93775,20.17489 -8.12469,2.44737 -55.98052,6.84692 -81.61317,3.75897 -4.8897,-0.5 -3.03411,-12.30852 -13.95543,-14.00216 C 140.83486,734.41525 69.224299,750.33502 18.30334,756.6 Z"
         style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccsccccccsc"
         inkscape:connector-curvature="0"
         id="path4168"
         d="m 202.30089,755.15533 c -1.27562,3.78866 -1.92477,9.05905 -4.62306,10.31729 -5.17933,1.7767 -10.93285,0.56235 -13.80762,5.00084 -4.18958,6.73156 -29.81038,25.81302 -47.58108,39.89277 -5.03872,3.99218 -13.55856,1.05692 -14.10863,-6.38647 l -0.0687,27.46438 12.61993,-0.6238 c 10.98748,-8.58886 41.27908,-37.35888 54.83042,-37.69627 10.77175,-0.85493 23.24146,-4.16643 26.20981,-11.33902 2.49974,-6.40744 -4.91423,-10.92996 -4.22864,-14.50428 1.06948,-4.17667 9.93603,-8.48091 12.90992,-8.51388 5.07625,-0.0563 -20.88362,-7.64511 -22.15235,-3.61156 z"
         style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccccccccc"
         inkscape:connector-curvature="0"
         id="path4170"
         d="M 44.513437,845.65221 C 96.654824,833.25125 219.71831,794.8025 237.75017,801.99454 c 22.3396,10.87105 63.54417,36.44588 64.11439,47.24215 0.22691,10.50433 0.0984,11.39774 -8.75819,19.87413 -13.17018,12.93087 -55.35363,77.18637 -61.49898,101.9057 -1.42211,6.27152 -0.44292,18.40166 -5.16715,23.4281 -6.61336,7.36518 -88.04586,-1.12959 -141.055824,-3.22941 -17.921009,-0.60114 -26.244151,-18.98182 0.533046,-19.54016 34.105198,-0.61411 92.367578,-2.02228 99.275418,-10.49309 6.23883,-7.95297 49.26453,-97.45139 49.36461,-107.09074 0.10549,-6.04568 -10.73558,-20.20992 -17.19309,-24.29398 -7.88187,-5.57042 -118.387423,28.03071 -167.577947,35.01891 z"
         style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path4172"
         d="m 22.789896,858.99298 c 13.914207,26.906 43.909736,76.58459 55.890318,124.51516 l 32.083776,-2.12639 C 102.82771,945.31445 83.44735,884.02726 64.641487,853.9626 30.77162,800.25976 -5.5262875,807.3679 22.789896,858.99298 Z"
         style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path4174"
         d="m 121.1644,783.92288 c -3.21262,-15.5446 -15.49505,-12.91539 -13.58709,3.49686 l 18.46383,193.39277 14.14333,-0.69937 z"
         style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path4176"
         d="m 189.6572,785.11604 -13.32425,0 -25.12742,191.8783 12.92012,0 z"
         style="fill:#19328c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"1s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou1.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou1.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.25274098"
     inkscape:cx="551.88588"
     inkscape:cy="606.25771"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g8091"
       transform="matrix(0.46690407,0,0,0.49873619,-17.749353,624.086)">
      <use
        
        
         transform="matrix(0.97502005,-0.2158284,0.22383997,0.97607119,-18.206062,79.986289)"
         id="use8027"
         xlink:href="#path8025"
         y="0"
         x="0" />
      <path
         sodipodi:nodetypes="cccccccccccccccccc"
         inkscape:connector-curvature="0"
         id="path8025"
         d="m 293.03997,673.85105 -6.48067,58.53848 c -18.40275,-0.94174 -31.30738,-7.63032 -43.6405,-11.53986 7.70013,14.94339 21.57114,24.31967 38.77427,33.5508 l -6.42343,41.35998 c -27.11146,-4.67369 -31.47135,-5.42045 -52.55588,-8.46154 18.84377,7.42934 25.5128,11.40601 38.50484,17.71047 -22.79997,3.80324 -29.59732,5.08388 -60.79326,12.18214 32.16241,-1.20595 43.32022,0.48929 68.43617,-0.40456 -10.85088,7.99164 -21.69369,10.38461 -31.45648,24.61746 25.97701,-5.2727 46.71847,-16.75397 60.16197,-23.19881 12.40823,-1.6986 21.10735,-4.15105 38.94182,-7.14634 -7.73033,-4.87842 -21.38125,-8.30987 -35.60437,-11.69578 l 5.44363,-43.43589 c 25.69643,-4.21215 43.57109,-16.88914 56.43054,-21.37992 -19.78916,3.63066 -31.81973,1.84949 -52.44499,-0.30165 l 6.77583,-59.51934 z"
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccc"
         inkscape:connector-curvature="0"
         id="path8021"
         d="m 205.67635,526.53066 c -27.32559,-13.07098 -34.65409,-22.28002 -43.3039,-31.74426 -17.85986,-19.09724 -35.06523,-29.80297 -55.83498,-43.12141 17.83824,15.23345 33.4055,27.66434 48.75249,54.20487 l -49.47455,-9.25046 c 36.11753,12.49428 63.51182,28.86559 81.97459,46.79468 z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccc"
         inkscape:connector-curvature="0"
         id="path8017"
         d="m 539.91824,586.55294 c 41.59994,-19.48882 77.21303,-43.33083 130.06757,-56.41007 -59.04898,22.3095 -91.22822,43.25599 -120.17137,73.5587 -3.11918,6.38776 0.4958,12.77551 1.65331,19.16327 L 522.44677,572.8905 c 6.8565,7.18264 12.1566,15.92846 17.47147,13.66244 z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:0.91780353px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccc"
         inkscape:connector-curvature="0"
         id="path8015"
         d="m 497.25019,523.92856 c 12.09173,17.88662 78.17632,-38.97952 112.69213,-61.32115 l -64.01112,52.28784 62.35408,-36.45957 c -36.53865,28.58695 -79.53222,47.52688 -93.7614,77.89866 z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:0.64343536px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         inkscape:original-d="m 229.02334,175.18539 c -6.92751,4.7356 -33.00026,0.4097 -43.62347,15.94675 -30.92642,45.23167 -55.47436,45.51635 5.96874,120.81035 -45.05712,28.69001 -104.489397,105.16731 28.70057,131.88373 116,19.92616 164.24705,5.40252 280.24705,4.00995 158.2902,-13.87304 93.27461,-116.70383 44.90551,-137.14283 C 601.29949,228.82217 540.38715,203.35067 509.31623,190.1438 563.73779,62.384306 429.15905,81.285988 380.40938,123.88078 330.82664,58.184147 296.21652,90.672835 271.25739,133.36829 c -22.17595,37.93451 -25.89501,31.28004 -42.23405,41.8171 z"
         inkscape:path-effect="#path-effect7963"
         sodipodi:nodetypes="csccccccsc"
         inkscape:connector-curvature="0"
         id="path7953"
         d="m 240.70089,192.26798 c -5.67953,0.86388 -9.1821,-0.21119 -11.53926,-2.04008 -3.21729,-2.17768 -5.54659,-4.74665 -7.63196,-7.38673 0,0 0,0 0,0 -5.23676,-6.59676 -9.59523,-13.51595 -15.62588,-19.88851 0,0 -10e-6,0 -10e-6,0 -1.17382,-1.27761 -2.40358,-2.53057 -3.70354,-3.74593 -3.26736,7.73183 -5.98695,15.86889 -6.39959,23.71338 -0.4028,7.70441 1.71927,15.09639 6.66954,19.79717 0.14605,-0.17923 0.90733,-0.56984 1.87003,-1.15247 1.54594,-0.84873 4.20347,-1.36146 7.66047,-2.04585 l 0,0 c 3.03493,-0.52764 6.94326,-0.87578 11.65425,-1.53734 l 0,0 c 1.86485,-0.23202 4.26903,-0.64586 7.17638,-1.36945 2.08153,-0.29227 5.4314,-1.66439 9.86957,-4.34419 z m -38.2307,10.4493 c -1.21489,-1.00129 -2.51887,-1.87335 -3.90405,-2.58291 -14.35666,-0.67905 -27.98978,-3.0628 -41.82624,-5.22165 4.0287,8.91126 7.91974,17.99294 12.15303,26.78571 2.52184,5.18672 5.24731,10.20399 8.52399,14.56216 1.04039,-1.59867 2.3262,-3.3565 3.76373,-5.31803 6.41913,-8.5238 14.00722,-17.54242 21.30069,-28.13123 l 1e-5,-1e-5 c -0.0458,-0.0181 -0.0474,-0.0496 -0.0112,-0.094 z m -25.05327,33.54331 c -3.09397,-0.59531 -6.20278,-1.07611 -9.35901,-1.3188 -7.32703,-0.64983 -15.21671,0.0851 -22.77706,4.02342 0,0 0,0 0,0 -4.42614,2.34291 -8.63311,5.65041 -12.42441,9.61043 3.69473,2.71482 7.60025,4.74926 11.5501,6.2297 11.55534,4.20417 23.052,4.10839 34.1382,3.34919 -1.94861,-3.96684 -3.50525,-7.34292 -4.04068,-10.17689 l 0,0 c -0.36296,-1.94676 -0.56332,-3.56347 -0.30521,-4.57564 l 0,0 c 0.0489,-0.99763 0.70519,-2.49111 1.60795,-4.38054 0.43096,-0.86769 0.97478,-1.7846 1.61012,-2.76087 z m 1.12782,21.89394 c 0,0 -0.002,0.002 -0.002,0.002 -9.64006,14.4054 -16.65276,26.37907 -21.99779,34.70832 15.4277,0.85486 29.68316,4.32341 43.17578,11.684 -8.4599,-15.65541 -16.1274,-30.80013 -21.17581,-46.39345 0,0 -5e-4,-0.001 -5e-4,-0.001 z m 21.17631,46.39446 c -1.96898,3.24558 -3.74997,5.89276 -5.40046,7.8689 -9.44143,1.93767 -19.12142,2.49135 -28.9585,1.75673 3.4761,6.55498 8.10672,17.36439 13.72795,31.9734 7.7428,-11.58021 16.46909,-22.31582 25.85157,-32.01967 -1.76256,-3.21936 -3.5078,-6.40983 -5.22056,-9.57936 z m -20.63101,41.59903 c -5.42554,-0.0316 -10.88393,-0.4875 -16.35914,-1.34888 -9.89803,-2.08642 -20.25708,-3.27247 -31.05309,-2.87188 2.28712,4.91333 4.82882,9.62023 7.65059,14.04496 5.69589,8.95469 12.63858,16.84534 20.73024,22.16697 0.80986,-2.91256 2.15603,-6.02336 3.78534,-9.27083 2.56376,-4.95611 6.05588,-9.80621 10.07181,-14.61175 l 0,0 c 1.67025,-2.74589 3.39632,-5.44951 5.17425,-8.10859 z m -19.0314,31.99117 c -3.7649,0.27514 -7.5358,0.89393 -11.27173,1.94701 -10.7497,2.92746 -21.37433,9.89515 -28.7614,21.38533 1.68019,0.73292 3.3705,1.38022 5.06458,1.93831 14.96227,4.72295 29.15127,1.95612 40.74901,-3.88639 -2.81074,-2.90469 -4.99851,-5.80183 -5.758,-8.51381 l 0,0 c -1.02028,-2.54233 -1.17176,-5.92787 -0.73381,-9.36586 0.13364,-1.13146 0.37671,-2.3011 0.71135,-3.50459 z m 5.78046,21.38426 c -0.72956,1.49899 -1.39263,3.05091 -1.98889,4.64467 -4.50508,12.57457 -4.97283,27.1274 -3.31151,41.83517 6.16198,-3.60588 11.85609,-7.68066 17.20205,-11.97563 6.86257,-5.50613 13.15487,-11.41 19.00345,-17.60258 -3.38686,-1.14946 -6.68548,-2.40597 -9.88111,-3.79431 l 0,0 c -7.93163,-3.5462 -14.95972,-7.22637 -19.84908,-11.92655 l 0,-1e-5 c -0.40174,-0.39343 -0.79405,-0.78716 -1.17491,-1.18076 z m 30.9051,16.90163 c 2.08149,15.25596 6.28971,30.73897 11.60686,46.07568 3.6539,-4.0912 7.16432,-8.25272 10.5611,-12.45886 6.35519,-8.0431 12.55784,-16.15273 18.58573,-24.32995 -4.54833,-0.70407 -9.09592,-1.44761 -13.6423,-2.22683 -9.44124,-1.93676 -18.5577,-4.15701 -27.11139,-7.06004 z m 40.75369,9.28687 c 5.20455,14.7617 10.85265,29.4876 17.03198,44.14947 9.23193,-12.74911 18.02212,-25.66775 26.29072,-38.75973 -14.43801,-1.31746 -28.88404,-3.15467 -43.3227,-5.38974 z m 43.3227,5.38974 c 4.66871,10.90437 9.65533,21.76117 14.99786,32.55816 1.71549,3.46933 3.46491,6.93077 5.24666,10.38424 8.23229,-13.38115 15.93663,-26.9274 23.20105,-40.60018 -8.86099,-0.15069 -17.69351,-0.48691 -26.48966,-1.04099 -5.65012,-0.34917 -11.3024,-0.78535 -16.95591,-1.30123 z m 43.44557,2.34222 c 7.29931,13.92046 15.11014,27.71728 23.32191,41.38458 4.29824,-8.26568 8.4521,-16.5669 12.48004,-24.89558 2.75362,-5.69011 5.45682,-11.39152 8.12346,-17.10238 -2.89209,0.11644 -5.77899,0.22167 -8.6608,0.31326 -11.79249,0.37436 -23.55343,0.49928 -35.26461,0.30012 z m 43.92541,-0.61338 c 8.1957,13.47552 16.69926,26.86126 25.27588,40.21155 6.55257,-14.17088 12.9634,-28.36366 19.43981,-42.55077 -15.04931,0.85791 -29.9501,1.74473 -44.71569,2.33922 z m 44.71569,-2.33922 c 3.61002,5.61287 7.22119,11.22732 10.81598,16.8474 4.93535,7.72643 9.88704,15.46347 14.84308,23.2131 6.48238,-14.13475 12.93769,-28.23513 19.41936,-42.29061 -11.76766,0.45857 -23.50787,1.03083 -35.21981,1.67675 -3.29335,0.18087 -6.5795,0.36643 -9.85861,0.55336 z m 45.07842,-2.23011 c 8.56438,13.46915 17.12372,26.98634 25.61614,40.56188 5.31204,-11.42464 10.65532,-22.81213 16.05917,-34.15671 1.12832,-2.60765 2.22163,-5.2201 3.27573,-7.83618 -1.21211,0.12328 -2.42822,0.24096 -3.648,0.35372 -13.80416,0.16192 -27.57228,0.54221 -41.30304,1.07729 z m 44.95104,-1.43101 c 8.04688,13.0309 17.10437,25.80054 27.85606,37.80682 5.63328,-15.03909 9.76312,-30.19178 11.68093,-45.25888 -12.32908,3.89773 -25.67133,6.04174 -39.53699,7.45206 z m 39.53699,-7.45206 c 0.53934,0.57862 1.08365,1.15456 1.63301,1.72774 10.14073,10.42376 21.68378,20.30969 35.84494,27.46056 0.763,-9.68213 0.45532,-19.16016 -1.32313,-28.09304 -1.24927,-6.19125 -3.27298,-12.07988 -6.06926,-17.31881 -1.30031,1.33067 -2.75365,2.63522 -4.30103,3.90295 -6.39021,5.1765 -15.12591,8.78818 -24.79527,12.00336 l 0,0 c -0.32902,0.10702 -0.65878,0.21276 -0.98926,0.31724 z m 30.08556,-16.22355 c 9.57254,4.32597 20.6293,6.3581 32.41399,4.11658 4.68137,-0.91847 9.40121,-2.49677 14.03417,-4.67727 -7.75132,-10.00598 -17.77866,-16.65283 -28.11236,-20.08159 -3.87397,-1.31272 -7.79998,-2.21632 -11.7375,-2.77458 0.2466,1.26483 0.41343,2.50559 0.48842,3.71727 l -10e-6,0 c 0.30054,4.36888 -0.21501,8.69651 -1.78427,12.03288 -0.97575,2.60039 -2.87171,5.17921 -5.30244,7.66671 z m 6.5983,-23.41686 c 9.33933,-5.62247 17.45333,-14.39327 23.98013,-24.58583 2.26051,-3.51709 4.34809,-7.21739 6.26387,-11.07005 -13.92112,-0.96217 -27.27535,0.64671 -39.75343,3.90021 -2.19696,0.57391 -4.36694,1.20215 -6.5072,1.88386 0.58201,0.67026 1.15506,1.34926 1.71893,2.03654 l 0,0 c 4.21216,5.19314 7.80523,10.63014 10.39245,16.2531 l 0,0 c 1.78235,3.98326 3.18452,7.88555 3.90525,11.58217 z m -16.01663,-29.87181 c 5.59472,-12.40101 9.33262,-26.16269 10.64933,-40.84803 0.29459,-1.2574 0.58903,-2.40865 0.87882,-3.45328 -12.41233,8.63045 -26.18942,11.07982 -39.04255,7.94217 -0.62978,-0.68398 -1.30235,-1.45098 -2.01851,-2.29898 -0.31194,1.05626 -0.62851,2.11264 -0.94935,3.1692 6.01185,6.95713 12.07034,15.07616 18.22238,23.77425 l 0,0 c 4.41577,3.46643 8.52243,7.41047 12.25988,11.71467 z m -29.53291,-38.65812 c 11.53568,-11.3518 25.6839,-16.46016 42.48931,-16.83506 -2.10621,-2.37657 -4.51412,-5.22104 -7.23652,-8.50229 -6.78677,-8.84052 -14.4343,-16.81058 -22.93197,-22.82929 -0.93527,3.6766 -2.30187,7.49049 -3.91618,11.40761 -1.80382,12.32066 -4.79385,24.53242 -8.40464,36.75903 z m 12.32082,-48.16664 c 5.71419,-0.87619 11.41433,-2.28824 17.02662,-4.42685 8.92539,-3.38422 17.65479,-8.70438 25.12765,-16.5137 -3.39234,-1.86879 -6.85121,-3.44952 -10.34228,-4.70387 -12.04026,-4.31297 -24.31471,-4.43087 -35.02993,-1.28151 1.80818,2.75063 3.16798,5.77003 3.86802,8.78074 l 0,1e-5 c 0.90651,3.44735 0.96759,7.52689 0.51409,11.81818 -0.2146,2.05708 -0.61504,4.16838 -1.16417,6.327 z m -3.21794,-26.92593 c 0,0 0.0259,-0.0329 0.0259,-0.0329 9.07068,-11.64655 12.76995,-27.81336 13.1994,-44.67655 -6.55709,2.49154 -12.68908,5.58953 -18.43781,9.00358 -7.63199,4.87534 -15.48198,8.18195 -23.94056,9.58291 4.11007,3.52328 8.02973,7.09368 11.7133,10.7877 l 10e-6,0 c 7.55337,4.81806 14.06157,9.69766 17.4326,15.32428 l 0,0 c 0,0 0.007,0.0109 0.007,0.0109 z M 513.5578,209.99229 c -0.0747,-11.09487 0.22363,-18.69804 2.05124,-22.38452 3.39952,-2.653 6.96103,-5.04526 10.67788,-7.20887 -5.85865,-2.43453 -14.39386,-8.62059 -25.83437,-17.09914 -0.81236,10.58966 -2.56122,21.36334 -4.70227,32.31118 6.21392,4.83606 12.18865,9.56468 17.80752,14.38135 z m -13.10525,-46.69253 c 8.25063,-5.54437 17.28028,-9.95573 26.99549,-13.68589 5.21001,-2.2955 10.392,-5.04642 15.3567,-8.42855 -13.21618,-9.35534 -27.91967,-15.59708 -42.035,-14.56603 -0.44094,0.0348 -0.88024,0.0766 -1.31768,0.12528 0.0413,0.10462 0.082,0.20981 0.12209,0.31553 2.10487,4.72283 1.95657,11.576 1.29048,19.27693 l 0,0 c 0.19158,5.59974 0.0258,11.2545 -0.41208,16.96273 z m -1.00049,-36.55519 c 5.449,-4.11121 10.1149,-9.70853 13.13689,-16.57579 3.05263,-6.93464 4.39724,-15.143077 3.78911,-23.751483 0,0 0,-2e-6 0,-2e-6 -0.0544,-0.741198 -0.12167,-1.485073 -0.20155,-2.231252 -16.82382,5.35875 -28.07484,16.976467 -35.69199,29.437487 5.23639,1.14671 9.8056,2.61666 12.72236,4.97061 1.45921,1.06436 2.74974,2.2812 3.72123,3.57622 l 0,0 c 0.99405,1.25794 1.83865,2.83726 2.52395,4.57421 z m -18.96754,-13.12104 c -0.10139,-0.53978 -0.20982,-1.07988 -0.3251,-1.62007 -3.13255,-14.125545 -10.70843,-27.911668 -20.40381,-40.579538 -1.46063,2.759733 -2.81252,5.543906 -4.06877,8.341868 -5.29271,11.743143 -8.97188,23.73853 -11.36579,35.74317 4.91822,-1.14798 9.83881,-1.99558 14.7302,-2.47967 7.44457,-0.68286 14.51404,-0.74736 20.83997,0.46626 l 0,0 c 0.19868,0.0422 0.39646,0.0849 0.5933,0.12798 z m -36.16347,1.88543 c -4.65369,-5.74446 -9.79634,-11.25473 -15.33176,-16.503385 -5.94597,-5.637726 -12.35047,-10.982898 -19.21202,-15.957555 -1.98288,12.163383 -2.81991,24.23363 -2.43746,36.00319 -0.40303,3.5175 -1.14206,6.95994 -2.20115,10.30927 2.20613,-0.61163 4.43686,-1.21009 6.68865,-1.79756 l 0,0 c 7.29317,-3.85231 15.04494,-7.00738 23.06323,-9.49767 l 0,0 c 3.13702,-0.96408 6.28328,-1.82169 9.43051,-2.55629 z m -39.18239,13.85152 c -13.72932,-8.13718 -21.38037,-14.30187 -23.54326,-16.94474 -0.1682,-1.44215 -0.30403,-2.88989 -0.40982,-4.34371 -3.74506,3.31957 -12.31654,7.22057 -23.41339,13.18137 7.05346,4.85689 14.03496,10.59944 21.18646,16.65126 8.12024,-3.18611 16.93399,-5.9808 26.18001,-8.54418 z m -47.36647,-8.10708 c -2.21737,-15.12872 -1.85955,-30.833208 -2.00793,-47.77528 -0.17548,0.109876 -0.3513,0.220335 -0.52748,0.33138 0,0 0,2e-6 0,2e-6 -11.20413,5.592131 -20.82689,13.65288 -25.76066,23.437284 0,0 0,2e-6 0,2e-6 -1.61005,3.213282 -2.72662,6.578562 -3.3492,9.982802 0.75053,-0.0298 1.49313,-0.002 2.20857,0.0859 l 0,0 c 2.28466,0.1505 5.0415,1.21255 8.05104,2.67962 l 0,0 c 7.35535,2.51309 14.40574,6.45203 21.38566,11.25827 z m -31.64527,-14.02381 c -1.44901,-5.19269 -4.21889,-10.178716 -8.1399,-14.500163 -6.44481,-7.083176 -15.66603,-12.123018 -26.00499,-15.521711 -0.3317,12.155989 2.62908,23.481334 6.56375,33.692644 1.53563,3.97296 3.254,7.82236 5.12084,11.56896 1.07365,-1.27528 2.16941,-2.50179 3.29183,-3.67431 5.74546,-5.53466 10.84281,-10.13861 15.44079,-10.95439 l 0,0 c 1.20523,-0.35266 2.47745,-0.5614 3.72768,-0.61103 z m -22.4603,15.23973 c -14.67788,-3.95981 -30.77218,-5.1066 -46.96509,-5.11605 2.11001,3.85976 4.30869,7.63379 6.56266,11.34231 5.99117,9.51418 11.92568,19.11365 16.98853,29.52534 3.259,-4.89986 6.16589,-9.74205 8.86869,-14.40959 4.67498,-7.86213 9.35142,-15.1728 14.54521,-21.34201 z m -23.4139,35.7516 c -6.54029,-0.14274 -13.03394,-0.51302 -19.38186,-1.29986 -9.49027,-1.15834 -18.88403,-3.27585 -25.4828,-8.81094 -1e-5,-1e-5 0.0273,0.0575 0.0272,0.0575 0,0 1e-5,0 1e-5,0 3.1593,5.65258 3.41024,12.68945 3.20951,19.78007 0,0 0,0 0,0 -0.30109,8.46437 -0.3383,16.9242 1.61329,24.6276 2.10648,-1.41085 4.89438,-2.79189 8.18427,-4.41415 3.37054,-1.54667 7.50503,-3.81284 12.10378,-7.1768 l 0,0 c 5.8573,-4.66015 10.24532,-9.5205 13.52095,-14.00284 l 0,0 c 2.21319,-2.92421 4.27142,-5.85256 6.20561,-8.76059 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccssc"
         inkscape:connector-curvature="0"
         id="path7923"
         d="M 86.22994,705.12826 C 113.37049,405.42899 543.65523,454.44072 544.39797,451.78891 l -49.41028,62.90362 c 0,0 70.69854,175.95815 147.19666,186.58365 44.36127,6.16173 -117.16805,-14.07003 -276.18142,-14.46648 -131.19571,-0.3271 -264.51051,18.82826 -279.77299,18.31856 z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:1.99338639;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:open="true"
         d="m 202.85915,575.39804 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88908,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="583.72504"
         sodipodi:cx="181.07133"
         sodipodi:type="arc"
         id="path7927"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path7931"
         sodipodi:type="arc"
         sodipodi:cx="225.15851"
         sodipodi:cy="545.87122"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 246.94632,537.54422 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         sodipodi:open="true"
         d="m 275.34915,578.31033 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70453 25.182831,16.605751 0 0 1 34.45814,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="586.63733"
         sodipodi:cx="253.56134"
         sodipodi:type="arc"
         id="path7933"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path7935"
         sodipodi:type="arc"
         sodipodi:cx="216.63756"
         sodipodi:cy="624.18146"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 238.42537,615.85446 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79364"
         sodipodi:open="true" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path7937"
         sodipodi:type="arc"
         sodipodi:cx="146.91129"
         sodipodi:cy="623.58832"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 168.6991,615.26132 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88908,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79364"
         sodipodi:open="true" />
      <path
         sodipodi:open="true"
         d="m 211.535,659.07291 a 25.182831,16.605751 0 0 1 -9.09474,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45814,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="667.3999"
         sodipodi:cx="189.74719"
         sodipodi:type="arc"
         id="path7939"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path7941"
         sodipodi:type="arc"
         sodipodi:cx="129.74399"
         sodipodi:cy="665.66467"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 151.5318,657.33768 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path7989"
         sodipodi:type="arc"
         sodipodi:cx="429.85782"
         sodipodi:cy="597.99426"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 451.64563,589.66727 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         sodipodi:open="true"
         d="m 476.05704,624.89731 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="633.2243"
         sodipodi:cx="454.26923"
         sodipodi:type="arc"
         id="path7991"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         sodipodi:open="true"
         d="m 506.57809,664.41617 a 25.182831,16.605751 0 0 1 -9.09474,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45814,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="672.74316"
         sodipodi:cx="484.79028"
         sodipodi:type="arc"
         id="path7993"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         sodipodi:open="true"
         d="m 515.92306,588.08548 a 25.182831,16.605751 0 0 1 -9.09474,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88908,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="596.41248"
         sodipodi:cx="494.13525"
         sodipodi:type="arc"
         id="path7995"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path7997"
         sodipodi:type="arc"
         sodipodi:cx="519.4668"
         sodipodi:cy="635.16718"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 541.25461,626.84018 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         sodipodi:open="true"
         d="m 569.76157,662.06406 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88908,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="670.39105"
         sodipodi:cx="547.97375"
         sodipodi:type="arc"
         id="path7999"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path8001"
         sodipodi:type="arc"
         sodipodi:cx="475.02408"
         sodipodi:cy="555.2851"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 496.81189,546.9581 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         sodipodi:open="true"
         d="m 436.46362,545.14792 a 25.182831,16.605751 0 0 1 -9.09474,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45814,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="553.47491"
         sodipodi:cx="414.67581"
         sodipodi:type="arc"
         id="path8003"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path8005"
         sodipodi:type="arc"
         sodipodi:cx="451.50793"
         sodipodi:cy="513.3667"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 473.29575,505.0397 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88908,-22.70453 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         sodipodi:open="true"
         d="m 440.3262,464.96359 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70453 25.182831,16.605751 0 0 1 34.45814,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="473.29059"
         sodipodi:cx="418.53839"
         sodipodi:type="arc"
         id="path8007"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path8009"
         sodipodi:type="arc"
         sodipodi:cx="482.06592"
         sodipodi:cy="475.40289"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 503.85373,467.0759 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79363"
         sodipodi:open="true" />
      <path
         inkscape:connector-curvature="0"
         id="path7987"
         d="m 272.88923,475.1926 c -23.28973,-1.45019 -52.549,-5.03538 -66.37946,-8.13356 -23.80127,-5.33177 -39.50915,-13.12812 -49.65138,-24.64366 -7.68025,-8.7202 -10.40351,-16.14546 -10.42732,-28.43114 -0.0405,-20.87821 11.86938,-46.58089 32.61194,-70.38003 2.29207,-2.62983 5.26904,-7.21219 6.6155,-10.18303 1.34644,-2.97083 4.56319,-9.41264 7.14833,-14.31514 l 4.70025,-8.91361 -12.4154,-14.1995 c -13.198,-15.09456 -18.15353,-21.62013 -24.33218,-32.04121 -2.11567,-3.56834 -5.07765,-8.3126 -6.58219,-10.54281 -3.33988,-4.95081 -2.4344,-9.48957 -2.41321,-14.06409 0.0431,-9.30687 7.59588,-20.25606 24.58738,-33.5875 7.62501,-5.98253 8.91602,-9.25479 31.46596,-10.29134 9.6285,-0.44259 18.80359,-1.22229 20.38909,-1.73267 7.20398,-2.31893 16.64297,-9.4717 22.40825,-16.98074 3.41125,-4.44303 5.41811,-8.03323 13.58968,-24.31149 12.25398,-24.41062 24.41902,-43.44901 34.59792,-54.146025 13.41667,-14.099574 22.92773,-17.625852 34.14141,-12.658117 12.4757,5.526827 23.26495,13.499307 36.97613,27.322772 l 6.94464,7.0015 6.3881,-2.44923 c 3.51345,-1.34708 10.60412,-3.69486 15.75704,-5.21729 5.15292,-1.52243 14.07145,-4.59491 19.81894,-6.82773 34.55595,-13.424615 70.30485,-12.414365 85.27073,2.40968 3.66145,3.62676 6.56018,9.77128 7.25041,15.36893 0.85677,6.94815 -0.86619,35.39656 -2.82422,46.63172 -3.09355,17.75081 -3.51933,15.41879 3.90134,21.36812 3.5319,2.83161 10.46933,9.03566 15.4165,13.78677 4.94717,4.75111 11.00188,10.25809 13.45491,12.23774 6.15742,4.96917 12.4123,12.72225 15.04282,18.64597 4.59435,10.34606 4.76897,19.58985 0.63286,33.50133 -1.36073,4.57671 -3.36722,12.51813 -4.45886,17.64763 -1.09165,5.12949 -3.33998,13.91136 -4.9963,19.51527 l -3.01147,10.18894 5.34643,6.43628 c 2.94055,3.53996 7.2735,9.04662 9.62876,12.23701 2.35528,3.19039 6.48982,7.6479 9.18787,9.90557 18.8579,15.77986 30.14007,37.73678 27.68333,53.87621 -1.19632,7.8591 -4.0021,13.58375 -10.36519,21.14817 -13.29012,15.79923 -37.7718,29.79522 -70.58767,40.35446 -7.28935,2.34551 -8.34795,2.46181 -28.82754,3.1668 -30.65178,1.05514 -38.50507,1.60724 -96.9326,6.81462 -17.44066,1.5544 -37.22355,3.20846 -43.96199,3.67569 -17.26825,1.19734 -59.08697,1.66196 -72.78954,0.80873 z"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         sodipodi:nodetypes="ssssssscsssssssssssscssssssssssssscssssssssss" />
      <path
         sodipodi:nodetypes="cssccccc"
         inkscape:connector-curvature="0"
         id="path8081"
         d="m 252.27553,375.37148 c -19.34213,2.58789 -54.33832,-16.75261 -57.16261,-15.57102 -1.34987,0.56474 43.65513,21.60765 39.92265,22.5103 -17.52397,4.23792 -36.72744,5.01925 -54.68202,7.06185 l 67,4 -68,16 c 23.16642,-0.68722 51.346,-7.4205 69.75,-2.75 -2.59545,-9.50202 0.19297,-22.67819 3.17198,-31.25113 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.94639778;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path8019"
         d="m 309.65846,440.69367 -106.31016,-38.30905 48.75786,37.23594 -57.76803,-9.47515 84.38428,53.22269 z"
         style="fill:#b93c3c;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         sodipodi:nodetypes="cccccc" />
      <g
         transform="matrix(0.88780381,0,0,0.88780381,29.358315,-235.49724)"
         id="g7969">
        <path
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path7899"
           sodipodi:type="arc"
           sodipodi:cx="-172.66649"
           sodipodi:cy="568.20685"
           sodipodi:rx="24.923891"
           sodipodi:ry="24.923891"
           sodipodi:start="6.2672801"
           sodipodi:end="4.733122"
           d="m -147.74575,567.81045 a 24.923891,24.923891 0 0 1 -15.22767,23.35821 24.923891,24.923891 0 0 1 -27.35941,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38508 24.923891,24.923891 0 0 1 23.43145,-15.11472"
           sodipodi:open="true"
           transform="matrix(0.61896829,-0.78541598,0.78541598,0.61896829,0,0)" />
        <path
           transform="matrix(0.78873557,-0.61473263,0.61473263,0.78873557,0,0)"
           sodipodi:open="true"
           d="M 65.341771,652.4128 A 24.923891,24.923891 0 0 1 50.114104,675.77101 24.923891,24.923891 0 0 1 22.754689,670.39046 24.923891,24.923891 0 0 1 17.506286,643.00539 24.923891,24.923891 0 0 1 40.937742,627.89067"
           sodipodi:end="4.733122"
           sodipodi:start="6.2672801"
           sodipodi:ry="24.923891"
           sodipodi:rx="24.923891"
           sodipodi:cy="652.8092"
           sodipodi:cx="40.421032"
           sodipodi:type="arc"
           id="path7901"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path7903"
           sodipodi:type="arc"
           sodipodi:cx="-119.08728"
           sodipodi:cy="672.25208"
           sodipodi:rx="24.923891"
           sodipodi:ry="24.923891"
           sodipodi:start="6.2672801"
           sodipodi:end="4.733122"
           d="m -94.166542,671.85567 a 24.923891,24.923891 0 0 1 -15.227668,23.35821 24.923891,24.923891 0 0 1 -27.35941,-5.38055 24.923891,24.923891 0 0 1 -5.24841,-27.38507 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:open="true"
           transform="matrix(0.71491107,-0.69921539,0.69921539,0.71491107,0,0)" />
        <path
           transform="matrix(0.40676759,-0.91353168,0.91353168,0.40676759,0,0)"
           sodipodi:open="true"
           d="m -371.41026,493.37145 a 24.923891,24.923891 0 0 1 -15.22766,23.35821 24.923891,24.923891 0 0 1 -27.35942,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38507 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:end="4.733122"
           sodipodi:start="6.2672801"
           sodipodi:ry="24.923891"
           sodipodi:rx="24.923891"
           sodipodi:cy="493.76785"
           sodipodi:cx="-396.33099"
           sodipodi:type="arc"
           id="path7905"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path7907"
           sodipodi:type="arc"
           sodipodi:cx="167.24132"
           sodipodi:cy="720.47974"
           sodipodi:rx="24.923891"
           sodipodi:ry="24.923891"
           sodipodi:start="6.2672801"
           sodipodi:end="4.733122"
           d="m 192.16206,720.08333 a 24.923891,24.923891 0 0 1 -15.22767,23.35822 24.923891,24.923891 0 0 1 -27.35942,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38508 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:open="true"
           transform="matrix(0.88801348,-0.45981742,0.45981742,0.88801348,0,0)" />
        <path
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path7909"
           sodipodi:type="arc"
           sodipodi:cx="-443.66245"
           sodipodi:cy="566.93292"
           sodipodi:rx="24.923891"
           sodipodi:ry="24.923891"
           sodipodi:start="6.2672801"
           sodipodi:end="4.733122"
           d="m -418.74171,566.53652 a 24.923891,24.923891 0 0 1 -15.22766,23.35821 24.923891,24.923891 0 0 1 -27.35942,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38507 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:open="true"
           transform="matrix(0.40676759,-0.91353168,0.91353168,0.40676759,0,0)" />
        <path
           transform="matrix(0.88801348,-0.45981742,0.45981742,0.88801348,0,0)"
           sodipodi:open="true"
           d="M 112.87851,760.10982 A 24.923891,24.923891 0 0 1 97.650843,783.46804 24.923891,24.923891 0 0 1 70.291428,778.08748 24.923891,24.923891 0 0 1 65.043025,750.70241 24.923891,24.923891 0 0 1 88.474482,735.58769"
           sodipodi:end="4.733122"
           sodipodi:start="6.2672801"
           sodipodi:ry="24.923891"
           sodipodi:rx="24.923891"
           sodipodi:cy="760.50623"
           sodipodi:cx="87.957771"
           sodipodi:type="arc"
           id="path7911"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           transform="matrix(0.71491107,-0.69921539,0.69921539,0.71491107,0,0)"
           sodipodi:open="true"
           d="m -198.85136,778.22609 a 24.923891,24.923891 0 0 1 -15.22766,23.35822 24.923891,24.923891 0 0 1 -27.35942,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38508 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:end="4.733122"
           sodipodi:start="6.2672801"
           sodipodi:ry="24.923891"
           sodipodi:rx="24.923891"
           sodipodi:cy="778.6225"
           sodipodi:cx="-223.77209"
           sodipodi:type="arc"
           id="path7913"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path7915"
           sodipodi:type="arc"
           sodipodi:cx="434.73495"
           sodipodi:cy="693.19507"
           sodipodi:rx="24.923891"
           sodipodi:ry="24.923891"
           sodipodi:start="6.2672801"
           sodipodi:end="4.733122"
           d="m 459.65569,692.79867 a 24.923891,24.923891 0 0 1 -15.22766,23.35821 24.923891,24.923891 0 0 1 -27.35942,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38508 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:open="true"
           transform="matrix(0.99252909,-0.12200825,0.12200825,0.99252909,0,0)" />
        <path
           transform="matrix(0.99252909,-0.12200825,0.12200825,0.99252909,0,0)"
           sodipodi:open="true"
           d="m 402.95129,763.65218 a 24.923891,24.923891 0 0 1 -15.22767,23.35821 24.923891,24.923891 0 0 1 -27.35942,-5.38055 24.923891,24.923891 0 0 1 -5.2484,-27.38507 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:end="4.733122"
           sodipodi:start="6.2672801"
           sodipodi:ry="24.923891"
           sodipodi:rx="24.923891"
           sodipodi:cy="764.04858"
           sodipodi:cx="378.03055"
           sodipodi:type="arc"
           id="path7917"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           transform="matrix(-6.3399233e-4,-0.9999998,0.9999998,-6.3399233e-4,0,0)"
           sodipodi:open="true"
           d="m -612.64371,243.54323 a 24.923891,24.923891 0 0 1 -15.22767,23.35822 24.923891,24.923891 0 0 1 -27.35942,-5.38056 24.923891,24.923891 0 0 1 -5.2484,-27.38507 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:end="4.733122"
           sodipodi:start="6.2672801"
           sodipodi:ry="24.923891"
           sodipodi:rx="24.923891"
           sodipodi:cy="243.93964"
           sodipodi:cx="-637.56445"
           sodipodi:type="arc"
           id="path7919"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:#ffffff;stroke-width:20.33586311;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="path7921"
           sodipodi:type="arc"
           sodipodi:cx="-713.42517"
           sodipodi:cy="294.9418"
           sodipodi:rx="24.923891"
           sodipodi:ry="24.923891"
           sodipodi:start="6.2672801"
           sodipodi:end="4.733122"
           d="m -688.50443,294.5454 a 24.923891,24.923891 0 0 1 -15.22767,23.35821 24.923891,24.923891 0 0 1 -27.35941,-5.38055 24.923891,24.923891 0 0 1 -5.24841,-27.38507 24.923891,24.923891 0 0 1 23.43146,-15.11472"
           sodipodi:open="true"
           transform="matrix(-6.3399233e-4,-0.9999998,0.9999998,-6.3399233e-4,0,0)" />
      </g>
      <path
         sodipodi:open="true"
         d="m 299.10693,504.0845 a 25.182831,16.605751 0 0 1 -9.09474,22.66909 25.182831,16.605751 0 0 1 -34.405,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70453 25.182831,16.605751 0 0 1 34.45814,5.79363"
         sodipodi:end="5.7459523"
         sodipodi:start="5.7579084"
         sodipodi:ry="16.605751"
         sodipodi:rx="25.182831"
         sodipodi:cy="512.4115"
         sodipodi:cx="277.31912"
         sodipodi:type="arc"
         id="path8150"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:2;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="path8146"
         sodipodi:type="arc"
         sodipodi:cx="311.26236"
         sodipodi:cy="551.79639"
         sodipodi:rx="25.182831"
         sodipodi:ry="16.605751"
         sodipodi:start="5.7579084"
         sodipodi:end="5.7459523"
         d="m 333.05017,543.46939 a 25.182831,16.605751 0 0 1 -9.09475,22.66909 25.182831,16.605751 0 0 1 -34.40499,-5.92936 25.182831,16.605751 0 0 1 8.88907,-22.70454 25.182831,16.605751 0 0 1 34.45813,5.79364"
         sodipodi:open="true" />
      <path
         sodipodi:nodetypes="cccccccccccccc"
         inkscape:connector-curvature="0"
         id="path7925"
         d="m 516.6047,688.24204 c -52.73589,2.36969 -102.16214,3.15366 -153.69599,-9.29222 -30.14644,-2.81527 -89.07209,17.62409 -151.22045,18.24043 -40.77621,3.18993 -40.91088,-13.40553 -26.3172,-36.91118 21.6316,-34.36529 113.74968,-82.14482 184.8731,-122.46139 -25.89498,-0.71656 -84.21557,-18.11282 -90.95992,-45.95651 -4.82704,-18.87696 0.79237,-41.4212 21.88956,-47.38363 22.09477,-6.31094 27.1909,-8.09312 60.14012,1.75727 11.04193,0.36884 22.92855,13.18244 32.77971,19.42192 13.94675,15.99279 30.80912,28.39192 34.48461,57.04448 -0.52618,12.67655 -1.42799,25.3531 -9.0749,38.02965 -16.12941,16.34856 -53.26996,41.05771 -74.23172,49.9612 19.95456,-6.36344 57.30205,-27.193 76.0467,-36.53897 28.67211,43.74252 61.86991,79.11787 95.28638,114.08895 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:12.43374538;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccc"
         inkscape:connector-curvature="0"
         id="path8013"
         d="m 332.15963,463.80859 c -2.0633,3.77372 -3.19342,5.76117 -2.60015,11.98626 3.66138,4.2961 6.3329,8.49978 14.35112,3.17095 5.13506,-4.42784 3.87465,-7.4185 3.5526,-11.45212 -2.71798,-6.08634 -2.35002,-8.14417 -6.01211,-9.09432 -4.44266,-0.76019 -6.93522,3.57784 -9.29146,5.38923 z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccc"
         inkscape:connector-curvature="0"
         id="path8023"
         d="m 170.11959,432.73817 c -49.27716,-3.16851 -76.028697,-15.68684 -124.124,-22.40093 36.497399,1.18227 53.092334,4.61328 90.22772,1.96764 -39.703398,-13.13911 -37.722801,-16.57217 -74.721624,-31.73972 47.699224,15.81458 57.729804,13.04365 115.251264,20.87043 z"
         style="fill:#004900;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccccccccc"
         inkscape:connector-curvature="0"
         id="path8057"
         d="m 355.43728,161.05216 c -8.89355,-6.98662 -9.59582,-15.37975 -1.27539,-25.32218 l -17.7238,15.69452 11.44314,-25.79002 -17.25235,20.0424 4.43279,-25.81406 -10.04286,22.07132 -5.61921,-22.20939 -3.11883,25.64685 -9.29225,-24.27231 c 5.17258,22.57552 -1.07092,29.41761 -7.16183,43.78379 19.50862,-8.96217 36.68135,-13.80182 55.61059,-3.83092 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:3.95740938;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.41672087;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 476.48209,173.90322 c -2.1574,-7.05563 -7.246,-12.85503 -15.31819,-17.37574 l 12.77181,-21.93415 -20.51716,16.32578 6.07857,-25.8384 -14.18221,23.83368 -3.36407,-25.43807 -7.17035,24.09418 -11.26618,-20.4065 c -0.99793,9.30908 1.38725,20.21345 -15.8521,21.86397 20.58957,2.74742 57.32318,15.55507 68.81988,24.87525 z"
         id="path8059"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccccccc" />
      <path
         sodipodi:nodetypes="cccccccccccc"
         inkscape:connector-curvature="0"
         id="path8061"
         d="m 468.70239,228.51104 c 6.87459,-13.44151 5.91482,-21.66011 8.36903,-32.1547 l 4.74658,29.19349 17.27719,-18.62208 -6.43914,23.71634 25.67798,-12.47712 -15.46014,17.04986 30.2516,1.62062 -23.55754,5.5891 22.5629,12.95349 c -11.79612,-5.72871 -23.72074,-6.61068 -36.55946,7.48809 2.45898,-28.94755 -12.40972,-30.98784 -26.869,-34.35709 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.35753345;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccccc"
         inkscape:connector-curvature="0"
         id="path8075"
         d="m 539.5,268.86221 -55.5,19 c 11.79595,6.04517 19.02545,9.12832 21.5,20.5 l 34,-10 -35,2 30,-14 -33,4 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.94639778;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccc"
         inkscape:connector-curvature="0"
         id="path8077"
         d="m 527.05025,357.01907 c 12.10273,4.57313 17.1662,8.93217 38.629,8.82526 l -50.629,7.17474 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.94639778;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:transform-center-y="7.0710128"
         inkscape:transform-center-x="-0.0021016771"
         sodipodi:nodetypes="cccccccc"
         inkscape:connector-curvature="0"
         id="path8079"
         d="m 489.61541,400.17122 c 20.37552,0.3348 42.00666,-6.0071 62.28389,-9.17153 -17.75955,9.74908 -37.23138,10.93676 -55.93136,15.9837 15.09552,4.81567 29.73523,9.95375 47.82741,15.23816 -18.98592,-1.6999 -37.03034,-6.47218 -51.69548,1.01845 11.11323,5.64014 21.6128,11.38255 41.44868,15.56891 l -72.09571,-2.8777 c 14.26357,-2.36474 31.60642,-26.677 28.16257,-35.75999 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.94639778;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccccccc"
         inkscape:connector-curvature="0"
         id="path8083"
         d="m 248.15944,267.46725 c 0.45929,-18.04338 0.92943,-20.10773 10.67463,-28.75265 -19.9149,-3.333 -23.14438,-12.72537 -34.54176,-20.81402 l 19.7066,25.42788 -45.77016,-13.98534 36.87041,19.07091 -49.58436,0 41.956,6.35697 c -2.17921,9.57005 -17.0522,9.67245 -32.42054,16.52812 16.21289,-3.92801 40.28411,-10.06734 53.10918,-3.83187 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.44686604;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccc"
         inkscape:connector-curvature="0"
         id="path8085"
         d="m 269.59008,177.54233 c 3.70529,16.00254 4.9128,30.00568 16.65694,43.04806 -7.19359,-3.15789 -8.71477,1.55941 -21.55605,3.34164 6.27848,-15.87721 1.22429,-31.05216 4.89911,-46.3897 z"
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.26737165;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:#ffffff;fill-opacity:1;fill-rule:evenodd;stroke:#ffffff;stroke-width:4.42769051;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 235.24568,293.61492 c -29.53986,-11.7129 -33.18373,-7.38013 -33.18373,-7.38013 l 30.45809,13.03992 z"
         id="path8152"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccc" />
      <path
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#000000;stroke-width:0.45772466px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 99.223295,628.41638 c -20.064961,4.57908 -36.050044,17.48558 -47.198424,26.26311 12.139723,-3.87822 26.670935,-8.53081 39.508313,-9.75625 6.820051,-0.17328 6.596694,-0.21618 5.800845,3.71308 l 15.657701,-38.79225 c -3.41945,3.5821 -8.36187,16.1038 -13.768435,18.57231 z"
         id="path8154"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
    </g>
  </g>
</svg>
`,"2s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou2.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou2.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.8101934"
     inkscape:cx="181.50416"
     inkscape:cy="206.80298"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="3832"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px"
     inkscape:snap-bbox="true">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="384.48931,-167.93786"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4384">
      <g
         transform="matrix(0.95915654,0,0,0.95915654,6.112593,26.80356)"
         id="g4337">
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4339"
           cx="135.05489"
           cy="680.42645"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           cy="680.42645"
           cx="164.01987"
           id="ellipse4341"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           cy="829.79926"
           cx="133.83817"
           id="ellipse4343"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="24.680117"
           ry="19.097431" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58492,818.81626 0,-134.67362"
           id="path4345"
           inkscape:connector-curvature="0" />
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4347"
           cx="165.64214"
           cy="829.79926"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4349"
           cx="160.99481"
           cy="748.83081"
           rx="32.786533"
           ry="25.370163" />
        <ellipse
           cy="748.83081"
           cx="138.32327"
           id="ellipse4351"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="32.786533"
           ry="25.370163" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58492,685.72349 0,37.84241"
           id="path4353"
           inkscape:connector-curvature="0" />
        <path
           inkscape:connector-curvature="0"
           id="path4355"
           d="m 149.58491,777.56215 0,46.73373"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <ellipse
           ry="8.6974945"
           rx="8.9920015"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4357"
           cx="149.58495"
           cy="683.5329" />
        <path
           inkscape:connector-curvature="0"
           id="path4359"
           d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           sodipodi:nodetypes="cc" />
        <path
           sodipodi:nodetypes="cc"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           id="path4361"
           inkscape:connector-curvature="0" />
      </g>
      <g
         id="g4320"
         transform="matrix(0.95915654,0,0,0.95915654,6.112593,235.23614)">
        <ellipse
           ry="19.097431"
           rx="24.680117"
           cy="680.42645"
           cx="135.05489"
           id="ellipse4322"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4324"
           cx="164.01987"
           cy="680.42645" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4326"
           cx="133.83817"
           cy="829.79926" />
        <path
           inkscape:connector-curvature="0"
           id="path4328"
           d="m 149.58492,818.81626 0,-134.67362"
           style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           cy="829.79926"
           cx="165.64214"
           id="ellipse4330"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="25.370163"
           rx="32.786533"
           cy="748.83081"
           cx="160.99481"
           id="ellipse4332"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="25.370163"
           rx="32.786533"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4334"
           cx="138.32327"
           cy="748.83081" />
        <path
           inkscape:connector-curvature="0"
           id="path4336"
           d="m 149.58492,685.72349 0,37.84241"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099221;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58491,777.56215 0,46.73373"
           id="path4338"
           inkscape:connector-curvature="0" />
        <ellipse
           cy="683.5329"
           cx="149.58495"
           id="ellipse4342"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="8.9920015"
           ry="8.6974945" />
        <path
           sodipodi:nodetypes="cc"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           id="path4344"
           inkscape:connector-curvature="0" />
        <path
           inkscape:connector-curvature="0"
           id="path4335"
           d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           sodipodi:nodetypes="cc" />
      </g>
    </g>
  </g>
</svg>
`,"3s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou3.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou3.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-2" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="2.0219279"
     inkscape:cx="150.10705"
     inkscape:cy="198.31103"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4337"
       transform="matrix(0.95915654,0,0,0.95915654,6.1125962,26.803561)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4339"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4341"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4343"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4345"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4347"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4349"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4351"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4353"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4355"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4357"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4359"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4361"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,235.23614)"
       id="g4271">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4273"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4275"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4277"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4279"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4281"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4283"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4285"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4287"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4289"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4291"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4293"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4295"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,235.23614)"
       id="g4297">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4299"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4301"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4303"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4305"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4307"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4309"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4311"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4313"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4315"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4317"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4319"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4321"
         inkscape:connector-curvature="0" />
    </g>
  </g>
</svg>
`,"4s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou4.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou4.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-5" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-0" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-8" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-1" />
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern-10" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="34.6399"
     inkscape:cy="91.561083"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4328">
      <g
         id="g4275"
         transform="matrix(0.95915654,0,0,0.95915654,-71.525046,235.23614)">
        <ellipse
           ry="19.097431"
           rx="24.680117"
           cy="680.42645"
           cx="135.05489"
           id="ellipse4277"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4280"
           cx="164.01987"
           cy="680.42645" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4282"
           cx="133.83817"
           cy="829.79926" />
        <path
           inkscape:connector-curvature="0"
           id="path4284"
           d="m 149.58492,818.81626 0,-134.67362"
           style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           cy="829.79926"
           cx="165.64214"
           id="ellipse4286"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="25.370163"
           rx="32.786533"
           cy="748.83081"
           cx="160.99481"
           id="ellipse4288"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="25.370163"
           rx="32.786533"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4290"
           cx="138.32327"
           cy="748.83081" />
        <path
           inkscape:connector-curvature="0"
           id="path4292"
           d="m 149.58492,685.72349 0,37.84241"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58491,777.56215 0,46.73373"
           id="path4294"
           inkscape:connector-curvature="0" />
        <ellipse
           cy="683.5329"
           cx="149.58495"
           id="ellipse4296"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="8.9920015"
           ry="8.6974945" />
        <path
           sodipodi:nodetypes="cc"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           id="path4298"
           inkscape:connector-curvature="0" />
        <path
           inkscape:connector-curvature="0"
           id="path4300"
           d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           sodipodi:nodetypes="cc" />
      </g>
      <g
         transform="matrix(0.95915654,0,0,0.95915654,83.750227,235.23614)"
         id="g4302">
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4304"
           cx="135.05489"
           cy="680.42645"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           cy="680.42645"
           cx="164.01987"
           id="ellipse4306"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           cy="829.79926"
           cx="133.83817"
           id="ellipse4308"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="24.680117"
           ry="19.097431" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58492,818.81626 0,-134.67362"
           id="path4310"
           inkscape:connector-curvature="0" />
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4312"
           cx="165.64214"
           cy="829.79926"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4314"
           cx="160.99481"
           cy="748.83081"
           rx="32.786533"
           ry="25.370163" />
        <ellipse
           cy="748.83081"
           cx="138.32327"
           id="ellipse4316"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="32.786533"
           ry="25.370163" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58492,685.72349 0,37.84241"
           id="path4318"
           inkscape:connector-curvature="0" />
        <path
           inkscape:connector-curvature="0"
           id="path4320"
           d="m 149.58491,777.56215 0,46.73373"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <ellipse
           ry="8.6974945"
           rx="8.9920015"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4322"
           cx="149.58495"
           cy="683.5329" />
        <path
           inkscape:connector-curvature="0"
           id="path4324"
           d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           sodipodi:nodetypes="cc" />
        <path
           sodipodi:nodetypes="cc"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           id="path4326"
           inkscape:connector-curvature="0" />
      </g>
      <g
         transform="matrix(0.95915654,0,0,0.95915654,-71.525046,26.803561)"
         id="g4337">
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4339"
           cx="135.05489"
           cy="680.42645"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           cy="680.42645"
           cx="164.01987"
           id="ellipse4341"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           cy="829.79926"
           cx="133.83817"
           id="ellipse4343"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="24.680117"
           ry="19.097431" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58492,818.81626 0,-134.67362"
           id="path4345"
           inkscape:connector-curvature="0" />
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4347"
           cx="165.64214"
           cy="829.79926"
           rx="24.680117"
           ry="19.097431" />
        <ellipse
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4349"
           cx="160.99481"
           cy="748.83081"
           rx="32.786533"
           ry="25.370163" />
        <ellipse
           cy="748.83081"
           cx="138.32327"
           id="ellipse4351"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="32.786533"
           ry="25.370163" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58492,685.72349 0,37.84241"
           id="path4353"
           inkscape:connector-curvature="0" />
        <path
           inkscape:connector-curvature="0"
           id="path4355"
           d="m 149.58491,777.56215 0,46.73373"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <ellipse
           ry="8.6974945"
           rx="8.9920015"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4357"
           cx="149.58495"
           cy="683.5329" />
        <path
           inkscape:connector-curvature="0"
           id="path4359"
           d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           sodipodi:nodetypes="cc" />
        <path
           sodipodi:nodetypes="cc"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           id="path4361"
           inkscape:connector-curvature="0" />
      </g>
      <g
         id="g4248"
         transform="matrix(0.95915654,0,0,0.95915654,83.750227,26.803561)">
        <ellipse
           ry="19.097431"
           rx="24.680117"
           cy="680.42645"
           cx="135.05489"
           id="ellipse4250"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4253"
           cx="164.01987"
           cy="680.42645" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4255"
           cx="133.83817"
           cy="829.79926" />
        <path
           inkscape:connector-curvature="0"
           id="path4257"
           d="m 149.58492,818.81626 0,-134.67362"
           style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <ellipse
           ry="19.097431"
           rx="24.680117"
           cy="829.79926"
           cx="165.64214"
           id="ellipse4259"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="25.370163"
           rx="32.786533"
           cy="748.83081"
           cx="160.99481"
           id="ellipse4261"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <ellipse
           ry="25.370163"
           rx="32.786533"
           style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="ellipse4263"
           cx="138.32327"
           cy="748.83081" />
        <path
           inkscape:connector-curvature="0"
           id="path4265"
           d="m 149.58492,685.72349 0,37.84241"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
        <path
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 149.58491,777.56215 0,46.73373"
           id="path4267"
           inkscape:connector-curvature="0" />
        <ellipse
           cy="683.5329"
           cx="149.58495"
           id="ellipse4269"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           rx="8.9920015"
           ry="8.6974945" />
        <path
           sodipodi:nodetypes="cc"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           id="path4271"
           inkscape:connector-curvature="0" />
        <path
           inkscape:connector-curvature="0"
           id="path4273"
           d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
           style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
           sodipodi:nodetypes="cc" />
      </g>
    </g>
  </g>
</svg>
`,"5sr":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou5-Dora.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Sou5-Dora.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="178.67439"
     inkscape:cy="235.18486"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150.35156,334.82897"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4335"
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,26.803561)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4337"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4340"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4342"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4344"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4346"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4348"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4350"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4352"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4354"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4356"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4358"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4360"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,26.803561)"
       id="g4437">
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4439"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4441"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4443"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4445"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4447"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4449"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4451"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4453"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4455"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4457"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4459"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4461"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,235.23614)"
       id="g4463">
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4465"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4467"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4469"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4471"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4473"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4475"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4477"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4479"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4481"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4483"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4485"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4487"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4489"
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,235.23614)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4491"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4493"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4495"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4497"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4499"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4501"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4503"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4505"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4507"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4509"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4511"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4513"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       id="g4515"
       transform="matrix(0.95915654,0,0,0.95915654,6.1125892,130.64527)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4517"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4519"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4521"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4523"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4525"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4527"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4529"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4531"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4533"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4535"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4537"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4539"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <circle
       style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="path4178"
       cx="150.35619"
       cy="998.50543"
       r="19.7831" />
  </g>
</svg>
`,"5s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou5.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou5.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="2.0219278"
     inkscape:cx="119.69996"
     inkscape:cy="211.14546"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,235.23614)"
       id="g4271">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4273"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4275"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4277"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4279"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4281"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4283"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4285"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4287"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4289"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4291"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4293"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4295"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,235.23614)"
       id="g4297">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4299"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4301"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4303"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4305"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4307"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4309"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4311"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4313"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4315"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4317"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4319"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4321"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4337"
       transform="matrix(0.95915654,0,0,0.95915654,6.112593,130.64527)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4339"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4341"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4343"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4345"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4347"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4349"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4351"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4353"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4355"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4357"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4359"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4361"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       id="g4307"
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,26.80356)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4310"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4312"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4315"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4317"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4319"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4321"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4323"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4325"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4327"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4329"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4331"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4333"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       id="g4335"
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,26.80356)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4337"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4340"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4342"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4344"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4346"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4348"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4350"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4352"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4354"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4356"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4358"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4360"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
  </g>
</svg>
`,"6s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou6.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou6.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="98.057169"
     inkscape:cy="225.77852"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4337"
       transform="matrix(0.95915654,0,0,0.95915654,6.1125924,26.803561)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4339"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4341"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4343"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4345"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4347"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4349"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4351"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4353"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4355"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4357"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4359"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4361"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,235.23614)"
       id="g4271">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4273"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4275"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4277"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4279"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4281"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4283"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4285"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4287"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4289"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4291"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4293"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4295"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,235.23614)"
       id="g4297">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4299"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4301"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4303"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4305"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4307"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4309"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4311"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4313"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4315"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4317"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4319"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4321"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4320"
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,26.803561)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4322"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4324"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4326"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4328"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4330"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4332"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4334"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4336"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4338"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4340"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4342"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4344"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       id="g4346"
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,26.803561)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4348"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4350"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4352"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4354"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4356"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4358"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4360"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4362"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4364"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4366"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4368"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4370"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,6.1125924,235.23614)"
       id="g4372">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4374"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4376"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4378"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4380"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4382"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4384"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4386"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4388"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4390"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4392"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4394"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4396"
         inkscape:connector-curvature="0" />
    </g>
  </g>
</svg>
`,"7s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou7.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou7.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="4.0438556"
     inkscape:cx="122.5938"
     inkscape:cy="312.22469"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       transform="matrix(1.5015,0,0,1.1493823,-450.65448,57.293456)"
       id="g4347">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4349"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4351"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4353"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4355"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4357"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4359"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4361"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4363"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4365"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4367"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-450.65448,324.27038)"
       id="g4281">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4283"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4285"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4287"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4289"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4291"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4293"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4295"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4297"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4299"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4301"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4303"
       transform="matrix(1.5015,0,0,1.1493823,-555.32292,324.27038)">
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4305"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4307"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4309"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4311"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4313"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4315"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4317"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4319"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4321"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4323"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-346.89034,324.27038)"
       id="g4325">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4327"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4329"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4331"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4333"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4335"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4337"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4339"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4341"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4343"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4345"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4374"
       transform="matrix(1.5015,0,0,1.1493823,-450.65448,192.31311)">
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4376"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4378"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4380"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4382"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4384"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4386"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4388"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332146;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4390"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4392"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332146;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4394"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332146;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-555.32292,192.31311)"
       id="g4237">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4239"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4241"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4243"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4245"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4247"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4249"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4251"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4253"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4255"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4257"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4259"
       transform="matrix(1.5015,0,0,1.1493823,-346.89034,192.31311)">
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4261"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4263"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4265"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4267"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4269"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4271"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4273"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4275"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4277"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4279"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"8s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou8.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou8.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.15851"
     inkscape:cx="-38.544934"
     inkscape:cy="215.19695"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4397">
      <ellipse
         ry="18.317427"
         rx="23.672094"
         cy="679.60535"
         cx="239.37206"
         id="ellipse4373"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4377"
         cx="238.20503"
         cy="822.87726" />
      <ellipse
         ry="24.333958"
         rx="31.447418"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4387"
         cx="242.50696"
         cy="745.21582" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4375"
         cx="267.15402"
         cy="679.60535" />
      <path
         inkscape:connector-curvature="0"
         id="path4379"
         d="m 253.30864,812.34285 0,-129.17308"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:52.75361252;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         cy="682.5849"
         cx="253.30867"
         id="ellipse4381"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.6247368"
         ry="8.3422585" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path6969"
         d="m 73.60657,743.8728 63.17505,-0.0606"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:34.7848587;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path6941"
         d="M 120.32426,780.51518 70.301832,767.60936"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:46.37981033;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path6937"
         d="M 66.18363,814.60579 170.64531,728.25421"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:45;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4273"
         cx="33.681141"
         cy="679.60535"
         rx="23.672094"
         ry="18.317427" />
      <ellipse
         cy="679.60535"
         cx="61.463093"
         id="ellipse4275"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="23.672094"
         ry="18.317427" />
      <ellipse
         cy="822.87726"
         cx="32.514114"
         id="ellipse4277"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="23.672094"
         ry="18.317427" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:52.75361252;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 47.617723,812.34285 0,-129.17308"
         id="path4279"
         inkscape:connector-curvature="0" />
      <ellipse
         ry="8.3422585"
         rx="8.6247368"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4291"
         cx="47.61776"
         cy="682.5849" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4281"
         cx="63.019115"
         cy="822.87726"
         rx="23.672094"
         ry="18.317427" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4283"
         cx="58.561592"
         cy="745.21582"
         rx="31.447418"
         ry="24.333958" />
      <ellipse
         cy="745.21582"
         cx="36.816036"
         id="ellipse4285"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="31.447418"
         ry="24.333958" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 47.617723,684.68605 0,36.2968"
         id="path4287"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4289"
         d="m 47.617713,772.7737 0,44.82497"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4293"
         d="m 38.643557,757.67222 c 4.055553,-6.17768 14.924169,-5.91141 18.337365,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 38.643557,740.13267 c 4.055553,-6.17769 14.924169,-5.91142 18.337365,0"
         id="path4295"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:34.7848587;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 226.62683,743.8728 -63.17505,-0.0606"
         id="path7498"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         cy="822.87726"
         cx="268.71002"
         id="ellipse4383"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:45;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="M 234.04977,814.60579 129.58809,728.25421"
         id="path7502"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <ellipse
         ry="24.333958"
         rx="31.447418"
         cy="745.21582"
         cx="264.2525"
         id="ellipse4385"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <rect
         ry="2.7680101"
         rx="1.9463779"
         y="708.35883"
         x="122.3024"
        
        
         id="rect6963"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:40;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path6975"
         d="m 115.50532,764.14365 6.98247,6.83299"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccc"
         inkscape:connector-curvature="0"
         id="path6979"
         d="m 136.49462,753.27628 13.38324,-10.62367 0.0495,-14.82294"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:46.37981033;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 179.90914,780.51518 50.02243,-12.90582"
         id="path7500"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         inkscape:connector-curvature="0"
         id="path4389"
         d="m 253.30864,684.68605 0,36.2968"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="M 230.51372,808.24494 197.53171,781.18239"
         id="path7532"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 253.30863,772.7737 0,44.82497"
         id="path4391"
         inkscape:connector-curvature="0" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 244.33447,757.67222 c 4.05555,-6.17768 14.92417,-5.91141 18.33737,0"
         id="path4393"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4395"
         d="m 244.33447,740.13267 c 4.05555,-6.17769 14.92417,-5.91142 18.33737,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 184.65325,764.06883 -6.83281,6.98263"
         id="path7534"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path7536"
         d="M 216.4052,761.71789 202.89881,750.89867"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 163.73878,753.27628 -13.38324,-10.62367 -0.0494,-14.82294"
         id="path7538"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccc" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path6971"
         d="m 69.71968,808.24494 32.98201,-27.06255"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="M 83.828203,761.71789 97.334592,750.89867"
         id="path6977"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       id="g4438"
       transform="matrix(1,0,0,-1,0,1704.7746)">
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4440"
         cx="239.37206"
         cy="679.60535"
         rx="23.672094"
         ry="18.317427" />
      <ellipse
         cy="822.87726"
         cx="238.20503"
         id="ellipse4442"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="23.672094"
         ry="18.317427" />
      <ellipse
         cy="745.21582"
         cx="242.50696"
         id="ellipse4444"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="31.447418"
         ry="24.333958" />
      <ellipse
         cy="679.60535"
         cx="267.15402"
         id="ellipse4446"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="23.672094"
         ry="18.317427" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:52.75361252;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 253.30864,812.34285 0,-129.17308"
         id="path4448"
         inkscape:connector-curvature="0" />
      <ellipse
         ry="8.3422585"
         rx="8.6247368"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4450"
         cx="253.30867"
         cy="682.5849" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:34.7848587;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 73.60657,743.8728 63.17505,-0.0606"
         id="path4452"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:46.37981033;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="M 120.32426,780.51518 70.301832,767.60936"
         id="path4454"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:45;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="M 66.18363,814.60579 170.64531,728.25421"
         id="path4456"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         cy="679.60535"
         cx="33.681141"
         id="ellipse4458"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4460"
         cx="61.463093"
         cy="679.60535" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4462"
         cx="32.514114"
         cy="822.87726" />
      <path
         inkscape:connector-curvature="0"
         id="path4464"
         d="m 47.617723,812.34285 0,-129.17308"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:52.75361252;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         cy="682.5849"
         cx="47.61776"
         id="ellipse4466"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.6247368"
         ry="8.3422585" />
      <ellipse
         ry="18.317427"
         rx="23.672094"
         cy="822.87726"
         cx="63.019115"
         id="ellipse4468"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="24.333958"
         rx="31.447418"
         cy="745.21582"
         cx="58.561592"
         id="ellipse4470"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="24.333958"
         rx="31.447418"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4472"
         cx="36.816036"
         cy="745.21582" />
      <path
         inkscape:connector-curvature="0"
         id="path4474"
         d="m 47.617723,684.68605 0,36.2968"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 47.617713,772.7737 0,44.82497"
         id="path4476"
         inkscape:connector-curvature="0" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 38.643557,757.67222 c 4.055553,-6.17768 14.924169,-5.91141 18.337365,0"
         id="path4478"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4480"
         d="m 38.643557,740.13267 c 4.055553,-6.17769 14.924169,-5.91142 18.337365,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path4482"
         d="m 226.62683,743.8728 -63.17505,-0.0606"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:34.7848587;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4484"
         cx="268.71002"
         cy="822.87726"
         rx="23.672094"
         ry="18.317427" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path4486"
         d="M 234.04977,814.60579 129.58809,728.25421"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:45;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4488"
         cx="264.2525"
         cy="745.21582"
         rx="31.447418"
         ry="24.333958" />
      <rect
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:40;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect4490"
        
        
         x="122.3024"
         y="708.35883"
         rx="1.9463779"
         ry="2.7680101" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 115.50532,764.14365 6.98247,6.83299"
         id="path4492"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 136.49462,753.27628 13.38324,-10.62367 0.0495,-14.82294"
         id="path4494"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccc" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path4496"
         d="m 179.90914,780.51518 50.02243,-12.90582"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:46.37981033;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 253.30864,684.68605 0,36.2968"
         id="path4498"
         inkscape:connector-curvature="0" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path4500"
         d="M 230.51372,808.24494 197.53171,781.18239"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4502"
         d="m 253.30863,772.7737 0,44.82497"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4504"
         d="m 244.33447,757.67222 c 4.05555,-6.17768 14.92417,-5.91141 18.33737,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 244.33447,740.13267 c 4.05555,-6.17769 14.92417,-5.91142 18.33737,0"
         id="path4506"
         inkscape:connector-curvature="0" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path4508"
         d="m 184.65325,764.06883 -6.83281,6.98263"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="M 216.4052,761.71789 202.89881,750.89867"
         id="path4510"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="ccc"
         inkscape:connector-curvature="0"
         id="path4512"
         d="m 163.73878,753.27628 -13.38324,-10.62367 -0.0494,-14.82294"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 69.71968,808.24494 32.98201,-27.06255"
         id="path4514"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         inkscape:connector-curvature="0"
         id="path4516"
         d="M 83.828203,761.71789 97.334592,750.89867"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"9s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou9.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Sou9.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="-56.043018"
     inkscape:cy="314.65006"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       transform="matrix(1.5015,0,0,1.1493823,-450.65448,57.293453)"
       id="g4347">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4349"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4351"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4353"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4355"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4357"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4359"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4361"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4363"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4365"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4367"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4303"
       transform="matrix(1.5015,0,0,1.1493823,-555.32292,324.27037)">
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4305"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4307"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4309"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4311"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4313"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4315"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4317"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4319"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4321"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4323"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-346.89034,324.27037)"
       id="g4325">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4327"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4329"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4331"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4333"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4335"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4337"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4339"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4341"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4343"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4345"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-555.32292,192.3131)"
       id="g4237">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4239"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4241"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4243"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4245"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4247"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4249"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4251"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4253"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4255"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4257"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4259"
       transform="matrix(1.5015,0,0,1.1493823,-346.89034,192.3131)">
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4261"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4263"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4265"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4267"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4269"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4271"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4273"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4275"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4277"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4279"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
    <g
       id="g4575"
       transform="matrix(1.5015,0,0,1.1493823,-450.65448,192.3131)">
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4577"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4579"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4581"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4583"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4585"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4587"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4589"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4591"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4593"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4595"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-450.65448,324.27037)"
       id="g4597">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4599"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4601"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4603"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4605"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#b93c3c;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4607"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4609"
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#b93c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4611"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4613"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4615"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4617"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4649"
       transform="matrix(1.5015,0,0,1.1493823,-555.32292,57.293453)">
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4651"
         cx="391.39905"
         cy="540.3103"
         r="13.404437" />
      <circle
         r="13.404437"
         cy="540.3103"
         cx="408.70386"
         id="circle4653"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="390.67212"
         id="circle4655"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 400.07987,609.92327 0,-71.43858"
         id="path4657"
         inkscape:connector-curvature="0" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4659"
         cx="409.67307"
         cy="612.17273"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4661"
         cx="406.89658"
         cy="574.39923"
         r="16.323313" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="393.35172"
         id="circle4663"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96958,537.20982 0,22.66136"
         id="path4665"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4667"
         d="m 399.96957,588.5505 0,21.47028"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         inkscape:connector-curvature="0"
         id="path4669"
         d="m 399.46822,574.02736 1.00819,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
    </g>
    <g
       transform="matrix(1.5015,0,0,1.1493823,-346.89034,57.293453)"
       id="g4671">
      <circle
         r="13.404437"
         cy="540.3103"
         cx="391.39905"
         id="circle4673"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4675"
         cx="408.70386"
         cy="540.3103"
         r="13.404437" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4677"
         cx="390.67212"
         cy="612.17273"
         r="13.404437" />
      <path
         inkscape:connector-curvature="0"
         id="path4679"
         d="m 400.07987,609.92327 0,-71.43858"
         style="fill:none;fill-rule:evenodd;stroke:#004900;stroke-width:29.07647514;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <circle
         r="13.404437"
         cy="612.17273"
         cx="409.67307"
         id="circle4681"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         r="16.323313"
         cy="574.39923"
         cx="406.89658"
         id="circle4683"
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <circle
         style="opacity:1;fill:#004900;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4685"
         cx="393.35172"
         cy="574.39923"
         r="16.323313" />
      <path
         inkscape:connector-curvature="0"
         id="path4687"
         d="m 399.96958,537.20982 0,22.66136"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.96957,588.5505 0,21.47028"
         id="path4689"
         inkscape:connector-curvature="0" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:8.37332153;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 399.46822,574.02736 1.00819,0"
         id="path4691"
         inkscape:connector-curvature="0" />
    </g>
  </g>
</svg>
`,E:`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Ton.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Ton.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.50548197"
     inkscape:cx="363.15718"
     inkscape:cy="458.34028"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4182"
       transform="matrix(0.90070005,0,0,0.97528977,22.480539,21.079387)">
      <path
         sodipodi:nodetypes="ccccccccccccccccccccccc"
         inkscape:connector-curvature="0"
         id="path4169"
         d="M 233.62512,755.29109 C 178.13441,766.77996 64.370638,790.79873 58.640429,785.71129 42.658218,774.8367 25.686157,781.4533 35.789288,796.11051 c 16.240064,27.58037 41.08272,52.95267 53.054176,82.84405 2.288277,4.33456 4.578233,5.00952 6.866511,4.97624 34.894055,0.59778 59.617015,-7.13696 92.959705,-10.00028 5.24648,-0.50585 12.18032,5.19713 16.07777,3.49424 3.45211,-1.88029 4.88869,-8.56661 5.7989,-12.75873 6.85855,-25.41545 28.05224,-47.67676 39.89617,-72.71328 1.50503,-2.99478 9.84505,-3.54971 9.78218,-6.44028 -1.21917,-10.81663 -17.3035,-30.7056 -26.59958,-30.22138 z m -23.13944,33.57895 c 12.46189,-0.30489 -0.39856,25.04533 -5.5016,26.11044 -5.68441,0.83505 -6.59045,-7.30247 -13.55667,-7.44593 -5.23392,0.54881 -4.04327,3.93106 -9.43905,5.54094 -32.92047,6.46479 -57.05235,8.76033 -91.40148,12.13597 -4.332557,0.10941 -15.065211,-17.22193 -9.013958,-18.63046 40.940618,-9.93723 109.784008,-17.44402 128.912758,-17.71096 z m -16.38918,50.02681 c 4.01395,6.86764 -2.84402,13.79985 -7.80917,14.56359 -30.19331,4.51502 -78.25363,12.14709 -82.00667,8.23091 -2.66879,-2.84434 -8.082537,-10.97917 -7.221842,-16.53793 11.136252,-3.71556 92.300252,-14.42426 97.037682,-6.25657 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccccccscccc"
         inkscape:connector-curvature="0"
         id="path4165"
         d="m 123.10112,656.13043 c -18.37065,-7.40086 -20.37189,5.83216 -17.90295,14.33847 6.2135,20.20143 14.42853,28.63207 24.53852,42.67785 1.80515,2.41778 3.80884,7.05879 3.50496,9.27068 l -2.80317,207.5969 c 0.0245,6.49844 -1.22391,27.05521 -5.33713,23.81363 -16.25707,-12.35739 -21.17086,-19.29273 -47.244611,-51.54601 -11.141011,-14.92882 -19.888636,-6.9441 -6.365572,12.59128 42.459393,54.62391 46.079123,59.83408 46.706613,115.10687 1.6238,29.1788 31.75855,24.6486 34.5721,6.1383 3.42954,-19.9718 -1.64734,-51.78318 -1.84988,-73.7618 -0.77433,-84.02764 -0.85925,-158.3381 12.84541,-239.53344 2.59385,-12.6668 22.99001,-23.06614 33.97476,-26.71095 4.86607,-1.61964 10.28698,-13.97417 0.90028,-14.45229 -25.17978,-2.64878 -52.28301,-15.55116 -75.53933,-25.52949 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="cccccc"
         inkscape:connector-curvature="0"
         id="path4167"
         d="m 90.058998,749.95518 c -28.498411,-1.30744 -7.40777,15.78722 -1.003352,15.35489 35.469434,-3.22236 93.934624,-11.66451 130.102704,-19.88981 2.50231,-0.0783 -0.19108,-17.18993 -17.33839,-18.91906 -5.21285,-0.25484 -4.31683,2.60115 -10.47943,4.83506 -27.63258,6.00227 -76.15552,20.22066 -101.281532,18.61892 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccc"
         inkscape:connector-curvature="0"
         id="path4178"
         d="M 101.07932,860.47476 C 81.038575,899.28326 62.611644,913.1005 1.9825743,939.28575 -21.477214,948.67859 -19.805065,962.93934 8.6272375,953.18208 85.796234,923.19531 101.97857,904.47533 114.56321,868.12926 Z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
      <path
         sodipodi:nodetypes="ccccccc"
         inkscape:connector-curvature="0"
         id="path4180"
         d="m 148.90381,861.23655 c 24.81185,28.93873 58.74403,106.95253 77.55965,95.36697 14.85312,-9.54021 33.94987,-10.66782 42.06647,-8.48169 10.62302,2.79871 11.13375,7.75465 17.07981,10.96967 15.63489,6.5882 12.36449,-2.02074 6.9373,-9.59456 -14.04344,-21.5996 -80.57109,-34.66827 -122.31035,-92.57081 z"
         style="fill:#142896;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" />
    </g>
  </g>
</svg>
`,"0m":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Man5-Dora.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Man5-Dora.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.4297189"
     inkscape:cx="179.88567"
     inkscape:cy="252.8135"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g6031"
       transform="translate(1034.3429,648.88567)">
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -973.89996,230.33452 c 101.51009,10.77646 95.08642,-15.76217 180.06105,-18.57669 18.88461,-2.74566 17.63524,-17.45957 -0.44879,-16.01707 -90.36841,2.24313 -58.77943,34.93937 -177.69318,16.09165 -28.99732,-5.17596 -24.26375,14.56756 -1.91908,18.50211 z"
         id="path6014"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -882.63587,187.13598 c 1.31149,-3.92427 -7.9972,-10.18109 -13.21935,-3.47239 -4.83118,7.47941 -6.26401,12.15587 -10.86019,17.41995 -1.95819,2.39301 -2.24128,7.72517 -0.21694,9.26756 7.56831,6.19882 6.72442,4.78418 11.49965,14.73834 l 20.53068,-1.76196 c -6.51994,-8.58948 -7.59705,-7.83487 -10.13116,-16.42437 -1.16483,-9.32704 -0.80813,-10.17567 2.39731,-19.76713 z"
         id="path6017"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="sccccccs" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -856.84278,149.17395 c -2.07241,-2.1647 -3.30957,-2.93207 -6.21343,0 l -15.39121,15.54072 c -1.34759,1.50043 -1.37158,3.7049 -0.26356,4.96996 l 9.16958,8.83355 c 4.35362,4.25203 5.46399,7.31404 2.89974,12.33575 -3.41972,9.61649 -5.98603,19.36747 -6.09656,24.7078 l 21.1809,-3.48708 c -3.95944,-9.81743 -3.87088,-12.35734 2.86523,-21.64648 5.55911,-6.32692 10.93309,-7.7739 5.99542,-16.41723 z"
         id="path6019"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="csccccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -834.43931,229.52922 c -2.37961,3.3e-4 -5.11545,0.35227 -7.98633,0.67579 -25.98124,4.23953 -44.63197,13.66167 -87.7832,12.14062 -5.84133,0.0296 -10.79709,3.10802 -10.5918,7.58984 0.41578,9.89302 7.70014,62.17673 8.30469,66.88868 0.85285,6.41756 9.79015,6.98803 10.96679,1.34179 0.21553,-1.03423 0.2214,-3.95763 0.0977,-7.99609 0.54781,-1.14352 1.39786,-2.02021 2.30859,-2.18359 l 56.98242,-5.0918 c 0.57955,0.007 1.01412,0.54453 1.32032,1.33008 -0.4418,20.09339 -0.68432,39.74185 -0.25391,46.62304 0.12358,1.52015 0.0514,3.12015 1.57031,4.48635 3.62687,3.0336 24.10801,1.3634 29.40821,-1.9473 1.46854,-1.0778 1.9999,-1.583 2.19726,-3.83006 -2.96095,-33.99173 -0.66802,-77.2174 1.60547,-110.87305 0.59643,-7.45293 -2.91134,-9.15502 -8.14648,-9.1543 z m -26.98047,16.76368 c 5.65219,-1.38928 2.02203,15.82857 0.0879,15.87695 l -58.82813,4.61133 c -5.12716,-1.0242 -4.8804,-13.06608 -1.42188,-13.22852 26.63848,-0.89616 35.74375,-3.18112 60.16211,-7.25976 z m -0.12891,29.15234 c 1.70565,-0.1306 2.01753,12.38455 -0.38086,12.58598 l -57.32031,4.79687 c -4.10107,-1.15065 -4.35221,-11.96168 -0.7793,-12.49218 z"
         id="path6021"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccscccccccccsccccccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -881.54242,246.32019 -19.23296,0.63476 -5.29091,90.55948 c -0.75114,10.29193 4.49195,20.6109 11.24113,20.7628 l 18.7421,-6.5465 z"
         id="path6023"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -942.63635,354.94033 0.84624,8.0276 c 54.24616,18.2836 41.63669,-8.576 96.89447,-15.15097 l -0.63469,-4.92845 c -42.24197,-2.98719 -64.35269,13.17942 -97.10602,12.05182 z"
         id="path6025"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -947.81557,295.50699 c -4.79067,-14.34919 -18.0476,-11.53087 -15.5695,0.88508 13.89663,43.4482 17.94173,67.49746 19.79888,70.85866 10.79236,18.2285 18.40255,13.4537 19.07847,-2.4339 0.11137,-4.8491 -14.23436,-43.49455 -23.30785,-69.30984 z"
         id="path6027"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m -990.50324,324.51204 c -8.35911,-1.30406 -7.60048,9.08422 -3.31653,11.28821 59.84729,26.46758 123.8383,-8.86564 177.96674,-5.69096 2.05825,0.36905 5.31253,1.56409 8.29438,3.20196 9.55669,5.04809 3.73393,19.51008 -0.97638,22.47878 -21.53617,9.8449 -16.51418,0.091 -40.67325,11.2497 -2.27744,1.4605 -8.25717,2.5966 -10.48172,1.6848 -8.94326,-2.9779 -16.30179,-8.1024 -22.15012,-12.6641 l -11.18847,5.6458 c 6.90544,8.6009 17.09988,12.6716 25.54075,16.3714 6.74722,3.2135 14.12571,2.8541 19.02891,3.0056 25.10802,0.3448 48.58902,-9.2971 60.40286,-10.7554 3.45799,-0.5303 7.48318,-6.5254 9.05051,-9.3096 9.69143,-15.8724 5.68822,-18.57024 -1.08565,-35.18913 -3.84938,-6.23214 -14.98024,-8.01151 -20.8275,-8.80455 -60.71768,-2.57426 -138.25414,18.81176 -189.58453,7.48749 z"
         id="path6029"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccccccccccccc" />
    </g>
    <g
       transform="matrix(0.88299048,0,0,0.98816906,13.1912,3.648797)"
       id="g5716">
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 120.27793,677.75471 c -7.8385,4.68172 -12.0636,5.88462 -15.28389,15.76934 -0.51069,1.5984 -0.89802,7.87579 1.45828,12.18189 20.42172,17.87538 -53.387678,56.71165 -79.77403,73.05807 -17.1742039,10.05844 -11.334569,21.33928 11.1927,11.45291 29.066321,-12.07831 80.29683,-26.14346 90.36555,-62.50547 0.0974,-3.98883 3.07469,-5.6683 7.33092,-6.61801 6.5143,-1.42607 0.79247,-27.81439 -2.95115,-34.69169 -3.74364,-6.87729 -9.80421,-9.95916 -12.33838,-8.64704 z"
         id="path5718"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 99.312378,743.71137 c 1.356942,4.63869 6.087332,16.55244 5.223812,21.54646 -3.97785,9.12562 -9.539572,12.59405 -11.761299,22.30502 -0.951932,5.34243 18.599519,21.85854 24.214479,19.2474 7.17233,-4.34808 10.98357,-13.34363 10.86546,-17.38476 -0.77831,-19.43426 -13.16868,-33.19626 -13.6164,-55.56915 z"
         id="path5720"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 128.52177,684.93516 c 46.74633,15.51693 66.83889,-16.27548 109.09748,-24.48481 12.11701,-2.3539 24.18389,2.32491 33.4817,4.16336 20.10867,4.51419 4.22782,31.73865 -11.12198,25.40944 -36.55813,-16.13394 -67.90561,8.15603 -102.60538,15.75123 -2.56615,0.41461 -4.13405,-2.70999 -5.7732,-4.83223 -5.31005,-6.87503 -13.32357,-1.20308 -19.85872,-3.34769 z"
         id="path5722"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cscccscc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 183.56224,691.9042 -31.76516,6.91681 c 9.85021,10.81876 -13.47538,53.48216 -25.20301,81.45969 l 28.50682,-3.84691 c 2.86369,-29.1146 21.0972,-56.08602 28.46135,-84.52959 z"
         id="path5724"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 114.44574,721.80311 c 15.7995,34.88788 69.14952,-14.0007 103.23262,-11.18105 l 0,19.4218 c -34.97155,-5.66636 -93.65276,35.5377 -107.63487,6.71249 z"
         id="path5726"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 210.5717,701.26372 c -2.78547,-0.46571 -10.74507,7.25181 -10.72639,10.04644 0.68906,24.48817 0.0838,48.42955 -2.14675,72.91772 -0.80858,11.46836 23.90304,2.38393 24.92498,-3.04856 3.43539,-25.63932 1.95788,-74.42799 -0.25808,-75.63801 -3.42035,-2.01357 -6.72578,-3.3576 -11.79376,-4.27759 z"
         id="path5728"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="cccccc" />
      <path
         style="fill:#d71e1e;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
         d="m 112.95927,778.81549 c 109.31862,-31.08806 166.20953,-8.22783 169.39263,-2.25441 2.14803,3.27773 1.68381,10.10921 -0.57407,13.53111 -18.85836,25.07686 -119.62667,-39.36712 -165.06741,10.70831 z"
         id="path5730"
         inkscape:connector-curvature="0"
         sodipodi:nodetypes="ccccc" />
    </g>
    <circle
       style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="path4178"
       cx="53.414371"
       cy="720.5152"
       r="19.7831" />
  </g>
</svg>
`,"0p":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Pin5-Dora.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Pin5-Dora.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 21.477976,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <pattern
       y="0"
       x="0"
      
      
       patternUnits="userSpaceOnUse"
       id="EMFhbasepattern" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.71485945"
     inkscape:cx="-162.1301"
     inkscape:cy="301.89564"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150,200"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g7942"
       transform="translate(4.6993944e-7,0.67748)">
      <g
         id="g7728"
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,245.01862)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7730"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7732"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7734"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7736"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7738"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7740"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,245.01862)"
         id="g7784">
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7786"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7788"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7790"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7792"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7794"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7796"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,288.32505,464.94965)"
         id="g7798">
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7800"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7802"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7804"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7806"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7808"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7810"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
      <g
         id="g7812"
         transform="matrix(1.0227353,0,0,1.0227353,432.4431,464.94965)">
        <circle
           r="59.759056"
           cy="485.48892"
           cx="-205.97568"
           id="circle7814"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="45.069927"
           cy="485.48892"
           cx="-205.97568"
           id="circle7816"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="38.810516"
           cy="485.48892"
           cx="-205.97568"
           id="circle7818"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <path
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1"
           id="path7820"
           sodipodi:type="arc"
           sodipodi:cx="-209.3149"
           sodipodi:cy="-484.05789"
           sodipodi:rx="21.068708"
           sodipodi:ry="21.06871"
           sodipodi:start="0"
           sodipodi:end="6.2079791"
           sodipodi:open="true"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)" />
        <circle
           r="7.8531041"
           cy="485.48892"
           cx="-205.97568"
           id="circle7822"
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           r="3.5860937"
           cy="485.48892"
           cx="-205.97568"
           id="circle7824"
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      </g>
      <g
         transform="matrix(1.0227353,0,0,1.0227353,360.63962,355.83523)"
         id="g7826">
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7828"
           cx="-205.97568"
           cy="485.48892"
           r="59.759056" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7830"
           cx="-205.97568"
           cy="485.48892"
           r="45.069927" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7832"
           cx="-205.97568"
           cy="485.48892"
           r="38.810516" />
        <path
           transform="matrix(0.99997625,-0.00689127,-0.00689127,-0.99997625,0,0)"
           d="m -188.24619,-484.05789 a 21.068708,21.06871 0 0 1 -20.6726,21.06498 21.068708,21.06871 0 0 1 -21.44992,-20.27292 21.068708,21.06871 0 0 1 19.86607,-21.82727 21.068708,21.06871 0 0 1 22.1969,19.4522"
           sodipodi:open="true"
           sodipodi:end="6.2079791"
           sodipodi:start="0"
           sodipodi:ry="21.06871"
           sodipodi:rx="21.068708"
           sodipodi:cy="-484.05789"
           sodipodi:cx="-209.3149"
           sodipodi:type="arc"
           id="path7834"
           style="opacity:1;fill:none;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#ffffff;stroke-width:6.60038757;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:13.20077528, 13.20077528;stroke-dashoffset:0;stroke-opacity:1" />
        <circle
           style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7836"
           cx="-205.97568"
           cy="485.48892"
           r="7.8531041" />
        <circle
           style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
           id="circle7838"
           cx="-205.97568"
           cy="485.48892"
           r="3.5860937" />
      </g>
    </g>
    <circle
       style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="path4178"
       cx="46.162918"
       cy="852.35797"
       r="19.7831" />
  </g>
</svg>
`,"0s":`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:osb="http://www.openswatchbook.org/uri/2009/osb"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  
  
   viewBox="0 0 300 400"
   id="svg2"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Sou5-Dora.svg"
   inkscape:export-filename="C:\\Users\\Fluffy\\Documents\\Projects\\ExtraRiichi\\Tiles\\Export\\Regular\\Sou5-Dora.png"
   inkscape:export-xdpi="180"
   inkscape:export-ydpi="180">
  <defs
     id="defs4">
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7963"
       is_visible="true"
       pattern="m -90.825902,-314.06958 23.03016,41.38503 13.798268,-41.38503 z"
       copytype="repeated_stretched"
       prop_scale="1"
       scale_y_rel="false"
       spacing="0"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <inkscape:path-effect
       effect="skeletal"
       id="path-effect7830"
       is_visible="true"
       pattern="M -12.828427,33.715729 -17,-11 l 9.0000001,0 z"
       copytype="repeated_stretched"
       prop_scale="-1"
       scale_y_rel="false"
       spacing="5.1"
       normal_offset="0"
       tang_offset="0"
       prop_units="false"
       vertical_pattern="false"
       fuse_tolerance="0"
       pattern-nodetypes="cccc" />
    <linearGradient
       id="linearGradient10055"
       osb:paint="solid">
      <stop
         style="stop-color:#000000;stop-opacity:1;"
         offset="0"
         id="stop10057" />
    </linearGradient>
    <marker
       inkscape:stockid="Arrow1Lstart"
       orient="auto"
       refY="0"
       refX="0"
       id="Arrow1Lstart"
       style="overflow:visible"
       inkscape:isstock="true">
      <path
         id="path4978"
         d="M 0,0 5,-5 -12.5,0 5,5 0,0 Z"
         style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:#ff5c00;stroke-width:1pt;stroke-opacity:1"
         transform="matrix(0.8,0,0,0.8,10,0)"
         inkscape:connector-curvature="0" />
    </marker>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7847">
      <ellipse
         style="opacity:1;fill:#822600;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse7849"
         cx="394"
         cy="552.36218"
         rx="349.49533"
         ry="216" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath4243-1">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle4245-4"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath7876">
      <circle
         style="opacity:1;fill:#000000;fill-opacity:0.29670332;fill-rule:nonzero;stroke:#000000;stroke-width:19.13299942;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="circle7878"
         cx="-264.65997"
         cy="-198.20665"
         r="293.95438" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14693">
      <rect
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="rect14695"
        
        
         x="-332.59583"
         y="383.49765"
         rx="1.2551664"
         ry="3.7514515"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
    <clipPath
       clipPathUnits="userSpaceOnUse"
       id="clipPath14952">
      <ellipse
         style="opacity:1;fill:#a53c3c;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:7;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse14954"
         cx="-271.34384"
         cy="647.25604"
         rx="69.057365"
         ry="116.91089"
         transform="matrix(0.99939083,-0.03489951,0.03489951,0.99939083,0,0)" />
    </clipPath>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#aeffff"
     bordercolor="#666666"
     borderopacity="1"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.0109639"
     inkscape:cx="178.67439"
     inkscape:cy="235.18486"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     inkscape:window-width="1920"
     inkscape:window-height="1017"
     inkscape:window-x="1912"
     inkscape:window-y="-8"
     inkscape:window-maximized="1"
     showguides="true"
     inkscape:guide-bbox="true"
     units="px">
    <inkscape:grid
       type="xygrid"
       id="grid4774"
       visible="true"
       dotted="false"
       color="#3f3fff"
       opacity="0.03921569"
       empcolor="#3f3fff"
       empopacity="0.07843137"
       enabled="false" />
    <sodipodi:guide
       position="150,200"
       orientation="0,1"
       id="guide8231"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
    <sodipodi:guide
       position="150.35156,334.82897"
       orientation="1,0"
       id="guide8233"
       inkscape:label=""
       inkscape:color="rgb(0,0,255)" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,-652.36216)">
    <g
       id="g4335"
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,26.803561)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4337"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4340"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4342"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4344"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4346"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4348"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4350"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4352"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4354"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4356"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4358"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4360"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,26.803561)"
       id="g4437">
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4439"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4441"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4443"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4445"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4447"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4449"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4451"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4453"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4455"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4457"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4459"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4461"
         inkscape:connector-curvature="0" />
    </g>
    <g
       transform="matrix(0.95915654,0,0,0.95915654,109.62944,235.23614)"
       id="g4463">
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4465"
         cx="135.05489"
         cy="680.42645"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="680.42645"
         cx="164.01987"
         id="ellipse4467"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         cy="829.79926"
         cx="133.83817"
         id="ellipse4469"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="24.680117"
         ry="19.097431" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,818.81626 0,-134.67362"
         id="path4471"
         inkscape:connector-curvature="0" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4473"
         cx="165.64214"
         cy="829.79926"
         rx="24.680117"
         ry="19.097431" />
      <ellipse
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4475"
         cx="160.99481"
         cy="748.83081"
         rx="32.786533"
         ry="25.370163" />
      <ellipse
         cy="748.83081"
         cx="138.32327"
         id="ellipse4477"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="32.786533"
         ry="25.370163" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58492,685.72349 0,37.84241"
         id="path4479"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4481"
         d="m 149.58491,777.56215 0,46.73373"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="8.6974945"
         rx="8.9920015"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4483"
         cx="149.58495"
         cy="683.5329" />
      <path
         inkscape:connector-curvature="0"
         id="path4485"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4487"
         inkscape:connector-curvature="0" />
    </g>
    <g
       id="g4489"
       transform="matrix(0.95915654,0,0,0.95915654,-93.20763,235.23614)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4491"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4493"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4495"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4497"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4499"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4501"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4503"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4505"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4507"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4509"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4511"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4513"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <g
       id="g4515"
       transform="matrix(0.95915654,0,0,0.95915654,6.1125892,130.64527)">
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="680.42645"
         cx="135.05489"
         id="ellipse4517"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4519"
         cx="164.01987"
         cy="680.42645" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4521"
         cx="133.83817"
         cy="829.79926" />
      <path
         inkscape:connector-curvature="0"
         id="path4523"
         d="m 149.58492,818.81626 0,-134.67362"
         style="fill:none;fill-rule:evenodd;stroke:#d71e1e;stroke-width:55.00000381;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <ellipse
         ry="19.097431"
         rx="24.680117"
         cy="829.79926"
         cx="165.64214"
         id="ellipse4525"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         cy="748.83081"
         cx="160.99481"
         id="ellipse4527"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1" />
      <ellipse
         ry="25.370163"
         rx="32.786533"
         style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:30;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         id="ellipse4529"
         cx="138.32327"
         cy="748.83081" />
      <path
         inkscape:connector-curvature="0"
         id="path4531"
         d="m 149.58492,685.72349 0,37.84241"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" />
      <path
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:12.51099205;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 149.58491,777.56215 0,46.73373"
         id="path4533"
         inkscape:connector-curvature="0" />
      <ellipse
         cy="683.5329"
         cx="149.58495"
         id="ellipse4535"
         style="opacity:1;fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
         rx="8.9920015"
         ry="8.6974945" />
      <path
         sodipodi:nodetypes="cc"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         d="m 140.22861,761.81761 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         id="path4537"
         inkscape:connector-curvature="0" />
      <path
         inkscape:connector-curvature="0"
         id="path4539"
         d="m 140.22861,743.53117 c 4.22825,-6.44075 15.55968,-6.16314 19.11822,0"
         style="fill:none;fill-rule:evenodd;stroke:#ffffff;stroke-width:10.42582703;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
         sodipodi:nodetypes="cc" />
    </g>
    <circle
       style="opacity:1;fill:#d71e1e;fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:10;stroke-linecap:butt;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="path4178"
       cx="150.35619"
       cy="998.50543"
       r="19.7831" />
  </g>
</svg>
`};var I={1:"Menzen Tsumo",2:"Riichi",3:"Chankan",4:"Rinshan Kaihou",5:"Haitei Raoyue",6:"Houtei Raoyui",7:"Haku",8:"Hatsu",9:"Chun",10:"Jikaze (Seat Wind)",11:"Bakaze (Prevalent Wind)",12:"Tanyao",13:"Iipeiko",14:"Pinfu",15:"Chanta",16:"Ittsu",17:"Sanshoku Doujun",18:"Double Riichi",19:"Sanshoku Doukou",20:"Sankantsu",21:"Toitoi",22:"San Ankou",23:"Shousangen",24:"Honroutou",25:"Chiitoitsu",26:"Junchan",27:"Honitsu",28:"Ryanpeiko",29:"Chinitsu",30:"Ippatsu",31:"Dora",32:"Akadora",33:"Ura Dora",35:"Tenhou",36:"Chiihou",37:"Daisangen",38:"Suuankou",39:"Tsuu Iisou",40:"Ryuu Iisou",41:"Chinroutou",42:"Kokushi Musou",43:"Shousuushii",44:"Suukantsu",45:"Chuuren Poutou",47:"Junsei Chuuren Poutou",48:"Suuankou Tanki",49:"Kokushi Musou 13-wait",50:"Daisuushii"},U=class{constructor(n){this.viewpoint=0;this.container=n;let e=document.getElementById("riichienv-viewer-style");e||(e=document.createElement("style"),e.id="riichienv-viewer-style",e.textContent=`
                .mahjong-board svg { width: 100%; height: 100%; display: block; }
                .mahjong-board .center-info svg { width: auto; height: 100%; }
                
                .tile-layer {
                    position: relative;
                    width: 100%; 
                    height: 100%;
                }
                .tile-bg, .tile-fg {
                    position: absolute;
                    top: 0; 
                    left: 0;
                    width: 100%;
                    height: 100%;
                }
                .tile-bg { 
                    z-index: 1; 
                    background-color: #fdfdfd; 
                    border-radius: 4px;
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                }
                .tile-fg { z-index: 2; }
                
                .active-player-highlight {
                    box-shadow: 0 0 20px 5px rgba(255, 230, 0, 0.4);
                    background-color: rgba(255, 230, 0, 0.05);
                    border-radius: 12px;
                }
                
                .river-grid {
                    display: grid;
                    grid-template-columns: repeat(6, 34px);
                    grid-template-rows: repeat(3, 46px);
                    gap: 2px;
                    width: 214px; 
                    height: 142px;
                    justify-content: start; 
                    align-content: start;
                }
                
                .tile-rotated {
                    transform: rotate(90deg) scale(0.9);
                    transform-origin: center center;
                }

                .call-overlay {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 3em;
                    font-weight: bold;
                    color: white;
                    text-shadow: 0 0 5px #ff0000, 0 0 10px #000;
                    padding: 10px 30px;
                    background: rgba(0,0,0,0.6);
                    border-radius: 10px;
                    border: 2px solid white;
                    z-index: 100;
                    pointer-events: none;
                    animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                }

                .modal-overlay {
                    position: absolute;
                    top: 50%; left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid #555;
                    z-index: 200;
                    text-align: center;
                    min-width: 400px;
                    box-shadow: 0 0 30px rgba(0,0,0,0.8);
                    animation: fadeIn 0.3s;
                }
                .modal-title { font-size: 1.5em; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #777; padding-bottom: 5px; }
                .modal-content { text-align: left; }
                .yaku-list { margin: 10px 0; padding-left: 20px; }
                .score-display { font-size: 1.2em; text-align: center; margin-top: 15px; font-weight: bold; background: #333; padding: 5px; border-radius: 4px;}

                .riichi-stick {
                    width: 100px;
                    height: 8px;
                    background: #eee;
                    border-radius: 4px;
                    border: 1px solid #999;
                    position: relative;
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                }
                .riichi-stick::after {
                    content: "";
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    width: 6px;
                    height: 6px;
                    background: red;
                    border-radius: 50%;
                    transform: translate(-50%, -50%);
                }

                .debug-panel {
                    position: fixed;
                    bottom: 10px;
                    right: 10px;
                    background: rgba(0,0,0,0.8);
                    color: #0f0;
                    font-family: monospace;
                    padding: 10px;
                    border-radius: 4px;
                    font-size: 11px;
                    max-width: 300px;
                    max-height: 200px;
                    overflow: auto;
                    z-index: 1000;
                    pointer-events: none;
                    white-space: pre-wrap;
                }

                @keyframes popIn {
                    from { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
                    to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
                }
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            `,document.head.appendChild(e)),this.styleElement=e}getTileHtml(n){if(n==="back")return`<div class="tile-layer"><div class="tile-bg">${j.back||j.blank}</div></div>`;let e=j.front||"",t=j[n];return t||(t=j.blank||""),`
            <div class="tile-layer">
                <div class="tile-bg">${e}</div>
                <div class="tile-fg">${t}</div>
            </div>
        `}render(n){this.container.innerHTML="";let e=document.createElement("div");e.className="mahjong-board",Object.assign(e.style,{position:"relative",width:"100%",aspectRatio:"1/1",maxWidth:"900px",margin:"0 auto",backgroundColor:"#2d5a27",borderRadius:"12px",overflow:"hidden",fontSize:"14px",color:"white",fontFamily:"sans-serif",boxSizing:"border-box"});let t=document.createElement("div");t.className="center-info",Object.assign(t.style,{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%, -50%)",backgroundColor:"#1a3317",padding:"15px",borderRadius:"8px",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",zIndex:"10",boxShadow:"0 4px 6px rgba(0,0,0,0.3)",minWidth:"120px"}),t.innerHTML=`
            <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">
                ${this.formatRound(n.round)} <span style="font-size:0.8em; opacity:0.8; margin-left:5px;">Honba: ${n.honba}</span>
            </div>
            <div style="margin-bottom: 8px;">Kyotaku: ${n.kyotaku}</div>
            <div style="display:flex; align-items: center; gap: 5px;">
                <span>Dora:</span>
                <div style="display:flex; gap:2px;">
                    ${n.doraMarkers.map(r=>`<div style="width:28px; height:38px;">${this.getTileHtml(r)}</div>`).join("")}
                </div>
            </div>
        `,e.appendChild(t);let o=[0,-90,180,90];if(n.players.forEach((r,m)=>{let h=(m-this.viewpoint+4)%4,k=document.createElement("div");Object.assign(k.style,{position:"absolute",top:"50%",left:"50%",width:"0",height:"0",display:"flex",justifyContent:"center",transform:`rotate(${o[h]}deg)`});let p=document.createElement("div");if(Object.assign(p.style,{width:"600px",height:"auto",minHeight:"220px",display:"flex",flexDirection:"column",alignItems:"center",transform:"translateY(140px)",transition:"background-color 0.3s",position:"relative"}),m===n.currentActor&&p.classList.add("active-player-highlight"),p.style.padding="10px",n.lastEvent&&n.lastEvent.actor===m){let i="",c=n.lastEvent.type;if(["chi","pon","kan","ankan","daiminkan","kakan"].includes(c)?(i=c.charAt(0).toUpperCase()+c.slice(1),c==="daiminkan"&&(i="Kan")):c==="hora"&&(i=n.lastEvent.target===n.lastEvent.actor?"Tsumo":"Ron"),i){let d=document.createElement("div");d.className="call-overlay",d.textContent=i,p.appendChild(d)}}if(r.waits&&r.waits.length>0){let i=document.createElement("div");Object.assign(i.style,{position:"absolute",bottom:"240px",left:"50%",transform:"translateX(-50%)",background:"rgba(0,0,0,0.8)",color:"#fff",padding:"5px 10px",borderRadius:"4px",fontSize:"14px",zIndex:"50",display:"flex",gap:"4px",alignItems:"center",pointerEvents:"none"}),i.innerHTML='<span style="margin-right:4px;">Wait:</span>',r.waits.forEach(c=>{i.innerHTML+=`<div style="width:24px; height:34px;">${this.getTileHtml(c)}</div>`}),p.appendChild(i)}let f=document.createElement("div"),v=["E","S","W","N"][r.wind],z=r.wind===0,g=`P${m}`;if(f.innerHTML=`
                <span style="font-weight:bold; font-size:1.1em; margin-right:8px; color: ${z?"#ff4d4d":"white"};">
                    ${v}
                </span>
                <span style="font-weight:bold; font-size:1.1em; margin-right:8px;">${g}</span>
                <span style="font-family:monospace; font-size:1.3em;">${r.score}</span>
            `,r.riichi&&(f.style.color="#ff6b6b",f.innerHTML+=' <span style="font-weight:bold; border:2px solid currentColor; padding:0 4px; border-radius:4px; font-size:0.9em; margin-left:8px;">REACH</span>'),f.style.marginBottom="8px",f.style.textShadow="1px 1px 2px #000",r.riichi){let i=document.createElement("div");i.className="riichi-stick",i.style.marginBottom="10px",p.appendChild(i)}let a=document.createElement("div");a.className="river-grid",r.discards.forEach(i=>{let c=document.createElement("div");if(c.style.width="34px",c.style.height="46px",c.style.position="relative",i.isRiichi){let d=document.createElement("div");d.style.width="100%",d.style.height="100%",d.className="tile-rotated",d.innerHTML=this.getTileHtml(i.tile),c.appendChild(d)}else c.innerHTML=this.getTileHtml(i.tile);i.isTsumogiri&&(c.style.filter="brightness(0.7)"),a.appendChild(c)});let s=document.createElement("div");s.style.display="flex",s.style.alignItems="flex-end",s.style.marginTop="15px",s.style.height="56px";let x=document.createElement("div");Object.assign(x.style,{display:"flex",width:"570px",height:"56px",justifyContent:"flex-start"});let D=(r.hand.length+r.melds.length*3)%3===2;if(r.hand.forEach((i,c)=>{let d=document.createElement("div");d.style.width="40px",d.style.height="56px",d.innerHTML=this.getTileHtml(i),D&&c===r.hand.length-1&&(d.style.marginLeft="10px"),x.appendChild(d)}),s.appendChild(x),r.melds.length>0){let i=document.createElement("div");i.style.display="flex",i.style.gap="8px",i.style.marginLeft="10px",r.melds.forEach(c=>{let d=document.createElement("div");d.style.display="flex",c.tiles.forEach(F=>{let P=document.createElement("div");P.style.width="34px",P.style.height="46px",P.innerHTML=this.getTileHtml(F),d.appendChild(P)}),i.appendChild(d)}),s.appendChild(i)}p.appendChild(f),p.appendChild(a),p.appendChild(s),k.appendChild(p),e.appendChild(k)}),n.lastEvent&&n.lastEvent.type==="end_kyoku"&&n.lastEvent.meta&&n.lastEvent.meta.results){let r=n.lastEvent.meta.results,m=document.createElement("div");m.className="modal-overlay",Object.assign(m.style,{maxHeight:"80vh",overflowY:"auto",width:"450px"});let h='<div class="modal-title">End Kyoku</div>';r.forEach((k,p)=>{let f=k.score,u=k.actor,v=k.target,z=u===v,g=f.yaku.map(a=>`<li>${I[a]||`Yaku ${a}`}</li>`).join("");h+=`
                    <div class="result-item" style="margin-bottom: 20px; ${p>0?"border-top: 1px solid #555; padding-top: 15px;":""}">
                        <div style="font-weight: bold; margin-bottom: 5px; color: #ffd700; font-size: 1.2em;">
                            P${u} ${z?"Tsumo":"Ron from P"+v}
                        </div>
                        <div class="modal-content">
                            <ul class="yaku-list" style="columns: 2;">${g}</ul>
                            <div style="display:flex; justify-content:space-between; margin-top:10px; font-weight:bold;">
                                <span>${f.han} Han</span>
                                <span>${f.fu} Fu</span>
                            </div>
                        </div>
                        <div class="score-display">
                            ${f.points} Points
                        </div>
                    </div>
                `}),m.innerHTML=h,e.appendChild(m)}let l=document.createElement("div");l.className="debug-panel";let w=n.lastEvent?JSON.stringify(n.lastEvent):"null";l.textContent=`Event: ${n.eventIndex} / ${n.totalEvents}
Last Event:
${w}`,e.appendChild(l),this.container.appendChild(e)}formatRound(n){return`${["East","South","West","North"][Math.floor(n/4)]} ${n%4+1}`}};var S=class{constructor(n,e){let t=document.getElementById(n);if(!t)throw new Error(`Container #${n} not found`);this.container=t,this.log=e,this.container.innerHTML="";let o=document.createElement("div");o.id=`${n}-board`,Object.assign(o.style,{display:"flex",justifyContent:"center",alignItems:"flex-start",width:"100%",overflow:"auto"}),this.container.appendChild(o),this.controlPanel=document.createElement("div"),Object.assign(this.controlPanel.style,{display:"flex",gap:"10px",justifyContent:"center",marginTop:"10px",padding:"10px",backgroundColor:"#f0f0f0",borderRadius:"4px"}),this.container.appendChild(this.controlPanel),this.gameState=new _(e),this.renderer=new U(o),this.initControls(),this.update()}initControls(){this.controlPanel.innerHTML="",this.controlPanel.style.flexDirection="column",this.controlPanel.style.alignItems="center";let n="display: flex; gap: 10px; align-items: center; justify-content: center; margin-bottom: 8px; flex-wrap: wrap;",e="padding: 6px 12px; cursor: pointer; border: 1px solid #ccc; background: white; border-radius: 4px; font-weight: bold; font-family: sans-serif; user-select: none;",t=(a,s,x)=>{let b=document.createElement("button");return b.textContent=a,b.style.cssText=e,b.onclick=s,x&&(b.title=x),b},o=document.createElement("div");o.style.cssText=n;let l=t("<< Turn",()=>{this.gameState.stepTurn(!1,this.renderer.viewpoint),this.update()},"Previous Turn (Viewpoint Player)"),w=t("< Step",()=>{this.gameState.stepBackward()&&this.update()}),r=t("Step >",()=>{this.gameState.stepForward()&&this.update()}),m=t("Turn >>",()=>{this.gameState.stepTurn(!0,this.renderer.viewpoint),this.update()},"Next Turn (Viewpoint Player)");o.appendChild(l),o.appendChild(w),o.appendChild(r),o.appendChild(m);let h=document.createElement("div");h.style.cssText=n;let k=document.createElement("select");k.style.padding="5px",k.style.borderRadius="4px";let p=this.gameState.getKyokuCheckpoints(),f=document.createElement("option");f.value="0",f.text="Start Game",k.appendChild(f),p.forEach(a=>{let s=document.createElement("option");s.value=a.index.toString(),s.text=`${this.renderer.formatRound(a.round)} - ${a.honba} Honba`,k.appendChild(s)}),k.onchange=()=>{this.gameState.jumpTo(parseInt(k.value)),this.update()},this.kyokuSelect=k;let u=document.createElement("select");u.style.padding="5px",u.style.borderRadius="4px",["Self (P0)","Right (P1)","Opp (P2)","Left (P3)"].forEach((a,s)=>{let x=document.createElement("option");x.value=s.toString(),x.text=a,u.appendChild(x)}),u.value="0",u.onchange=()=>{this.renderer.viewpoint=parseInt(u.value),this.update()},this.viewpointSelect=u;let v=t("Prev Kyoku",()=>{let a=0;for(let s of p)if(s.index<this.gameState.cursor-5)a=s.index;else break;this.gameState.jumpTo(a),this.update()}),z=t("Next Kyoku",()=>{let a=p.find(s=>s.index>this.gameState.cursor);a&&(this.gameState.jumpTo(a.index),this.update())});h.appendChild(v),h.appendChild(k),h.appendChild(z);let g=document.createElement("span");g.style.margin="0 10px",g.style.borderLeft="1px solid #999",g.style.height="20px",h.appendChild(g),h.appendChild(u),this.controlPanel.appendChild(o),this.controlPanel.appendChild(h)}update(){this.renderer.render(this.gameState.current);let n=this.gameState.getKyokuCheckpoints(),e=0;for(let t of n)if(t.index<=this.gameState.cursor)e=t.index;else break;this.kyokuSelect&&(this.kyokuSelect.value=e.toString())}};window.RiichiEnvViewer=S;return T(A);})();
