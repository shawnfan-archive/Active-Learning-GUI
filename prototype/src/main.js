import Vue from 'vue'
import App from './App.vue'
import Router from './router.js'
import VueKonva from 'vue-konva'
import Vuetify from 'vuetify'
import 'vuetify/dist/vuetify.min.css'
import VueRouter from 'vue-router'
import '@mdi/font/css/materialdesignicons.css'

Vue.config.productionTip = false

Vue.use(VueKonva)
Vue.use(Vuetify, {
  iconfont: 'mdi'
})

new Vue({
  router: Router,
  vuetify: new Vuetify(),
  template: "<App/>",
  render: h => h(App)
}).$mount('#app')
